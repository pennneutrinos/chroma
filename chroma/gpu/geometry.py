import numpy as np
import pycuda.driver as cuda
from pycuda import gpuarray as ga
from pycuda import characterize

from chroma.geometry import standard_wavelengths
from chroma.gpu.tools import get_cu_module, get_cu_source, cuda_options, \
    chunk_iterator, format_array, format_size, to_uint3, to_float3, \
    make_gpu_struct, GPUFuncs, mapped_empty, Mapped

from chroma.log import logger

class GPUGeometry(object):
    def __init__(self, geometry, wavelengths=None, times=None, print_usage=False, min_free_gpu_mem=300e6):
        if wavelengths is None:
            wavelengths = standard_wavelengths

        try:
            wavelength_step = np.unique(np.diff(wavelengths)).item()
        except ValueError:
            raise ValueError('wavelengths must be equally spaced apart.')
            
        if times is None:
            time_step = 0.05
            times = np.arange(0,1000,time_step)
        else:
            try:
                time_step = np.unique(np.diff(times)).item()
            except ValueError:
                raise ValueError('times must be equally spaced apart.')

        geometry_source = get_cu_source('geometry_types.h')
        material_struct_size = characterize.sizeof('Material', geometry_source)
        surface_struct_size = characterize.sizeof('Surface', geometry_source)
        dichroicprops_struct_size = characterize.sizeof('DichroicProps', geometry_source)
        geometry_struct_size = characterize.sizeof('Geometry', geometry_source)

        self.material_data = []
        self.material_ptrs = []

        def interp_material_property(wavelengths, property):
            # note that it is essential that the material properties be
            # interpolated linearly. this fact is used in the propagation
            # code to guarantee that probabilities still sum to one.
            return np.interp(wavelengths, property[:,0], property[:,1]).astype(np.float32)

        for i in range(len(geometry.unique_materials)):
            material = geometry.unique_materials[i]

            if material is None:
                raise Exception('one or more triangles is missing a material.')

            refractive_index = interp_material_property(wavelengths, material.refractive_index)
            refractive_index_gpu = ga.to_gpu(refractive_index)
            absorption_length = interp_material_property(wavelengths, material.absorption_length)
            absorption_length_gpu = ga.to_gpu(absorption_length)
            scattering_length = interp_material_property(wavelengths, material.scattering_length)
            scattering_length_gpu = ga.to_gpu(scattering_length)
            num_comp = len(material.comp_reemission_prob)
            comp_reemission_prob_gpu = [ga.to_gpu(interp_material_property(wavelengths, component)) for component in material.comp_reemission_prob]
            self.material_data.append(comp_reemission_prob_gpu)
            comp_reemission_prob_gpu = np.uint64(0) if len(comp_reemission_prob_gpu) == 0 else make_gpu_struct(8*len(comp_reemission_prob_gpu), comp_reemission_prob_gpu)
            assert num_comp == len(material.comp_reemission_wvl_cdf), 'component arrays must be same length'
            comp_reemission_wvl_cdf_gpu = [ga.to_gpu(interp_material_property(wavelengths, component)) for component in material.comp_reemission_wvl_cdf]
            self.material_data.append(comp_reemission_wvl_cdf_gpu)
            comp_reemission_wvl_cdf_gpu = np.uint64(0) if len(comp_reemission_wvl_cdf_gpu) == 0 else make_gpu_struct(8*len(comp_reemission_wvl_cdf_gpu), comp_reemission_wvl_cdf_gpu)
            assert num_comp == len(material.comp_reemission_time_cdf), 'component arrays must be same length'
            comp_reemission_time_cdf_gpu = [ga.to_gpu(interp_material_property(times, component)) for component in material.comp_reemission_time_cdf]
            self.material_data.append(comp_reemission_time_cdf_gpu)
            comp_reemission_time_cdf_gpu = np.uint64(0) if len(comp_reemission_time_cdf_gpu) == 0 else make_gpu_struct(8*len(comp_reemission_time_cdf_gpu), comp_reemission_time_cdf_gpu)
            assert num_comp == len(material.comp_absorption_length), 'component arrays must be same length'
            comp_absorption_length_gpu = [ga.to_gpu(interp_material_property(wavelengths, component)) for component in material.comp_absorption_length]
            self.material_data.append(comp_absorption_length_gpu)
            comp_absorption_length_gpu = np.uint64(0) if len(comp_absorption_length_gpu) == 0 else make_gpu_struct(8*len(comp_absorption_length_gpu), comp_absorption_length_gpu)

            self.material_data.append(refractive_index_gpu)
            self.material_data.append(absorption_length_gpu)
            self.material_data.append(scattering_length_gpu)
            self.material_data.append(comp_reemission_prob_gpu)
            self.material_data.append(comp_reemission_wvl_cdf_gpu)
            self.material_data.append(comp_reemission_time_cdf_gpu)
            self.material_data.append(comp_absorption_length_gpu)

            material_gpu = \
                make_gpu_struct(material_struct_size,
                                [refractive_index_gpu, absorption_length_gpu,
                                 scattering_length_gpu,
                                 comp_reemission_prob_gpu,
                                 comp_reemission_wvl_cdf_gpu,
                                 comp_reemission_time_cdf_gpu,
                                 comp_absorption_length_gpu,
                                 np.uint32(num_comp),
                                 np.uint32(len(wavelengths)),
                                 np.float32(wavelength_step),
                                 np.float32(wavelengths[0]),
                                 np.uint32(len(times)),
                                 np.float32(time_step),
                                 np.float32(times[0])])

            self.material_ptrs.append(material_gpu)

        self.material_pointer_array = \
            make_gpu_struct(8*len(self.material_ptrs), self.material_ptrs)

        self.surface_data = []
        self.surface_ptrs = []

        for i in range(len(geometry.unique_surfaces)):
            surface = geometry.unique_surfaces[i]

            if surface is None:
                # need something to copy to the surface array struct
                # that is the same size as a 64-bit pointer.
                # this pointer will never be used by the simulation.
                self.surface_ptrs.append(np.uint64(0))
                continue

            detect = interp_material_property(wavelengths, surface.detect)
            detect_gpu = ga.to_gpu(detect)
            absorb = interp_material_property(wavelengths, surface.absorb)
            absorb_gpu = ga.to_gpu(absorb)
            reemit = interp_material_property(wavelengths, surface.reemit)
            reemit_gpu = ga.to_gpu(reemit)
            reflect_diffuse = interp_material_property(wavelengths, surface.reflect_diffuse)
            reflect_diffuse_gpu = ga.to_gpu(reflect_diffuse)
            reflect_specular = interp_material_property(wavelengths, surface.reflect_specular)
            reflect_specular_gpu = ga.to_gpu(reflect_specular)
            eta = interp_material_property(wavelengths, surface.eta)
            eta_gpu = ga.to_gpu(eta)
            k = interp_material_property(wavelengths, surface.k)
            k_gpu = ga.to_gpu(k)
            reemission_cdf = interp_material_property(wavelengths, surface.reemission_cdf)
            reemission_cdf_gpu = ga.to_gpu(reemission_cdf)
            
            if surface.dichroic_props:
                props = surface.dichroic_props
                transmit_pointers = []
                reflect_pointers = []
                angles_gpu = ga.to_gpu(np.asarray(props.angles,dtype=np.float32))
                self.surface_data.append(angles_gpu)
                
                for i,angle in enumerate(props.angles):
                    dichroic_reflect = interp_material_property(wavelengths, props.dichroic_reflect[i])
                    dichroic_reflect_gpu = ga.to_gpu(dichroic_reflect)
                    self.surface_data.append(dichroic_reflect_gpu)
                    reflect_pointers.append(dichroic_reflect_gpu)
                    
                    dichroic_transmit = interp_material_property(wavelengths, props.dichroic_transmit[i])
                    dichroic_transmit_gpu = ga.to_gpu(dichroic_transmit)
                    self.surface_data.append(dichroic_transmit_gpu)
                    transmit_pointers.append(dichroic_transmit_gpu)
                
                reflect_arr_gpu = make_gpu_struct(8*len(reflect_pointers),reflect_pointers)
                self.surface_data.append(reflect_arr_gpu)
                transmit_arr_gpu = make_gpu_struct(8*len(transmit_pointers), transmit_pointers)
                self.surface_data.append(transmit_arr_gpu)
                dichroic_props = make_gpu_struct(dichroicprops_struct_size,[angles_gpu,reflect_arr_gpu,transmit_arr_gpu,np.uint32(len(props.angles))])
            else:
                dichroic_props = np.uint64(0) #NULL
            
            

            self.surface_data.append(detect_gpu)
            self.surface_data.append(absorb_gpu)
            self.surface_data.append(reemit_gpu)
            self.surface_data.append(reflect_diffuse_gpu)
            self.surface_data.append(reflect_specular_gpu)
            self.surface_data.append(reemission_cdf_gpu)
            self.surface_data.append(eta_gpu)
            self.surface_data.append(k_gpu)
            self.surface_data.append(dichroic_props)
            
            surface_gpu = \
                make_gpu_struct(surface_struct_size,
                                [detect_gpu, absorb_gpu, reemit_gpu,
                                 reflect_diffuse_gpu,reflect_specular_gpu,
                                 eta_gpu, k_gpu, reemission_cdf_gpu,
                                 dichroic_props,
                                 np.uint32(surface.model),
                                 np.uint32(len(wavelengths)),
                                 np.uint32(surface.transmissive),
                                 np.float32(wavelength_step),
                                 np.float32(wavelengths[0]),
                                 np.float32(surface.thickness)])

            self.surface_ptrs.append(surface_gpu)

        self.surface_pointer_array = \
            make_gpu_struct(8*len(self.surface_ptrs), self.surface_ptrs)

        self.vertices = mapped_empty(shape=len(geometry.mesh.vertices),
                                     dtype=ga.vec.float3,
                                     write_combined=True)
        self.triangles = mapped_empty(shape=len(geometry.mesh.triangles),
                                      dtype=ga.vec.uint3,
                                      write_combined=True)
        self.vertices[:] = to_float3(geometry.mesh.vertices)
        self.triangles[:] = to_uint3(geometry.mesh.triangles)
        
        self.world_origin = ga.vec.make_float3(*geometry.bvh.world_coords.world_origin)
        self.world_scale = np.float32(geometry.bvh.world_coords.world_scale)

        material_codes = (((geometry.inner_material_index & 0xff) << 24) |
                          ((geometry.outer_material_index & 0xff) << 16) |
                          ((geometry.surface_index & 0xff) << 8)).astype(np.uint32)
        self.material_codes = ga.to_gpu(material_codes)
        colors = geometry.colors.astype(np.uint32)
        self.colors = ga.to_gpu(colors)
        self.solid_id_map = ga.to_gpu(geometry.solid_id.astype(np.uint32))

        # Limit memory usage by splitting BVH into on and off-GPU parts
        gpu_free, gpu_total = cuda.mem_get_info()
        node_array_usage = geometry.bvh.nodes.nbytes

        # Figure out how many elements we can fit on the GPU,
        # but no fewer than 100 elements, and no more than the number of actual nodes
        n_nodes = len(geometry.bvh.nodes)
        split_index = min(
            max(int((gpu_free - min_free_gpu_mem) / geometry.bvh.nodes.itemsize),100),
            n_nodes
            )
        
        self.nodes = ga.to_gpu(geometry.bvh.nodes[:split_index])
        n_extra = max(1, (n_nodes - split_index)) # forbid zero size
        self.extra_nodes = mapped_empty(shape=n_extra,
                                        dtype=geometry.bvh.nodes.dtype,
                                        write_combined=True)
        if split_index < n_nodes:
            logger.info('Splitting BVH between GPU and CPU memory at node %d' % split_index)
            self.extra_nodes[:] = geometry.bvh.nodes[split_index:]

        # See if there is enough memory to put the and/ortriangles back on the GPU
        gpu_free, gpu_total = cuda.mem_get_info()
        if self.triangles.nbytes < (gpu_free - min_free_gpu_mem):
            self.triangles = ga.to_gpu(self.triangles)
            logger.info('Optimization: Sufficient memory to move triangles onto GPU')

        gpu_free, gpu_total = cuda.mem_get_info()
        if self.vertices.nbytes < (gpu_free - min_free_gpu_mem):
            self.vertices = ga.to_gpu(self.vertices)
            logger.info('Optimization: Sufficient memory to move vertices onto GPU')

        self.gpudata = make_gpu_struct(geometry_struct_size,
                                       [Mapped(self.vertices), 
                                        Mapped(self.triangles),
                                        self.material_codes,
                                        self.colors, self.nodes,
                                        Mapped(self.extra_nodes),
                                        self.material_pointer_array,
                                        self.surface_pointer_array,
                                        self.world_origin,
                                        self.world_scale,
                                        np.int32(len(self.nodes))])

        self.geometry = geometry

        if print_usage:
            self.print_device_usage()
        logger.info(self.device_usage_str())

    def device_usage_str(self):
        '''Returns a formatted string displaying the memory usage.'''
        s = 'device usage:\n'
        s += '-'*10 + '\n'
        #s += format_array('vertices', self.vertices) + '\n'
        #s += format_array('triangles', self.triangles) + '\n'
        s += format_array('nodes', self.nodes) + '\n'
        s += '%-15s %6s %6s' % ('total', '', format_size(self.nodes.nbytes)) + '\n'
        s += '-'*10 + '\n'
        free, total = cuda.mem_get_info()
        s += '%-15s %6s %6s' % ('device total', '', format_size(total)) + '\n'
        s += '%-15s %6s %6s' % ('device used', '', format_size(total-free)) + '\n'
        s += '%-15s %6s %6s' % ('device free', '', format_size(free)) + '\n'
        return s

    def print_device_usage(self):
        print(self.device_usage_str())
        print() 

    def reset_colors(self):
        self.colors.set_async(self.geometry.colors.astype(np.uint32))

    def color_solids(self, solid_hit, colors, nblocks_per_thread=64,
                     max_blocks=1024):
        solid_hit_gpu = ga.to_gpu(np.array(solid_hit, dtype=np.bool))
        solid_colors_gpu = ga.to_gpu(np.array(colors, dtype=np.uint32))

        module = get_cu_module('mesh.h', options=cuda_options)
        color_solids = module.get_function('color_solids')

        for first_triangle, triangles_this_round, blocks in \
                chunk_iterator(self.triangles.size, nblocks_per_thread,
                               max_blocks):
            color_solids(np.int32(first_triangle),
                         np.int32(triangles_this_round), self.solid_id_map,
                         solid_hit_gpu, solid_colors_gpu, self.gpudata,
                         block=(nblocks_per_thread,1,1), 
                         grid=(blocks,1))

