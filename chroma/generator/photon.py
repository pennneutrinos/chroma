import multiprocessing
import numpy as np
import threading
import zmq
import uuid
import os, sys
sdir = os.path.dirname(__file__)
sys.path.append(sdir)
import g4gen

class G4GeneratorProcess(multiprocessing.Process):
    def __init__(self, idnum, material, vertex_socket_address, photon_socket_address, seed=None, tracking=False):
        multiprocessing.Process.__init__(self)

        self.idnum = idnum
        self.material = material
        self.vertex_socket_address = vertex_socket_address
        self.photon_socket_address = photon_socket_address
        self.seed = seed
        self.tracking = tracking
        self.daemon = True

    def run(self):
        print("process do run 1")
        gen = g4gen.G4Generator(self.material, seed=self.seed)
        print("process do run 2")
        context = zmq.Context()
        print("process do run 3")
        vertex_socket = context.socket(zmq.PULL)
        vertex_socket.connect(self.vertex_socket_address)
        print("process do run 4")
        photon_socket = context.socket(zmq.PUSH)
        photon_socket.connect(self.photon_socket_address)


        print("process do send")
        # Signal with the photon socket that we are online
        # and ready for messages.
        photon_socket.send(b'READY')

        while True:
            ev = vertex_socket.recv_pyobj()
            if self.tracking:
                ev.vertices,ev.photons_beg,ev.photon_parent_trackids = gen.generate_photons(ev.vertices,tracking=self.tracking)
            else:
                ev.vertices,ev.photons_beg = gen.generate_photons(ev.vertices,tracking=self.tracking)
            photon_socket.send_pyobj(ev)

def partition(num, partitions):
    """Generator that returns num//partitions, with the last item including
    the remainder.

    Useful for partitioning a number into mostly equal parts while preserving
    the sum.

    Examples:
        >>> list(partition(800, 3))
        [266, 266, 268]
        >>> sum(list(partition(800, 3)))
        800
    """
    step = num // partitions
    for i in range(partitions):
        if i < partitions - 1:
            yield step
        else:
            yield step + (num % partitions)

def vertex_sender(vertex_iterator, zmq_context, vertex_address, pgen):
    vertex_socket = zmq_context.socket(zmq.PUSH)
    vertex_socket.bind(vertex_address)
    length = 0
    for vertex in vertex_iterator:
        pgen.semaphore.acquire()
        vertex_socket.send_pyobj(vertex)
        length += 1     
    pgen.length = length

class G4ParallelGenerator(object):
    def __init__(self, nprocesses, material, base_seed=None, tracking=False):
        self.material = material
        if base_seed is None:
            base_seed = np.random.randint(100000000)
        base_address = 'ipc:///tmp/chroma_'+str(uuid.uuid4())
        self.vertex_address = base_address + '.vertex'
        self.photon_address = base_address + '.photon'
        self.processes = [ G4GeneratorProcess(i, material, self.vertex_address, self.photon_address, seed=base_seed + i, tracking=tracking) for i in range(nprocesses) ]
        
        for p in self.processes:
            p.start()

        self.zmq_context = zmq.Context()
        self.photon_socket = self.zmq_context.socket(zmq.PULL)
        self.photon_socket.bind(self.photon_address)

        self.processes_initialized = False
    
    def generate_events(self, vertex_iterator):
        print('M> G4PG Gen')
        if not self.processes_initialized:
            # Verify everyone is running and connected to avoid
            # sending all the events to one client.
            print(f'Init proc now count is {len(self.processes)}')
            for i in range(len(self.processes)):
                print(f'Chillin for msg {i}')
                msg = self.photon_socket.recv()
                print(f'Got msg: {msg}')
                assert msg == b'READY'
            self.processes_initialized = True
            
        print('boink')
        #let it get ahead, but not too far ahead
        self.semaphore = threading.Semaphore(2*len(self.processes)) 
        self.processed = 0
        self.length = -1
        sender_thread = threading.Thread(target=vertex_sender, args=(vertex_iterator, self.zmq_context, self.vertex_address, self))
        sender_thread.start()
        p = zmq.Poller()
        p.register(self.photon_socket, zmq.POLLIN)

        while self.length < 0 or self.processed < self.length:
            msgs = dict(p.poll(5000)) 
            if self.photon_socket in msgs and msgs[self.photon_socket] == zmq.POLLIN:
                yield self.photon_socket.recv_pyobj()
                self.semaphore.release()
                self.processed += 1
