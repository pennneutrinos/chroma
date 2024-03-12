from __future__ import annotations

import json
from pathlib import Path
from copy import deepcopy
from typing import Callable, Optional

from chroma.log import logger


class RatDBParser:
    def __init__(self, ratdb_path: Path | str, run_number=None, merge=True):
        """
        Args:
            ratdb_path: file path of the dumped RATDB file, in JSON format.
            run_number: Run number for extracting run-specific entries. Default is None and uses generic entries only.
            merge: If True, merge all planes in the database with appropriate overriding priorities, guarantee '
                uniqueness of each entry.
        """
        self.ratdb_path = Path(ratdb_path)
        self.run_number = run_number
        with open(self.ratdb_path, 'r') as f:
            self.entries = json.load(f)
        self.db = None
        if merge:
            self.merge_all_planes()
            self.db = self.create_db()
        else:
            logger.warn("Data base is not merged, no hash table is created since entry uniqueness is not guaranteed.")

    def get_entries_for_plane(self, plane_name: str, run_number=None):
        """
        Args:
            plane_name: Name of the plane to extract entries for. "default", "user", or "run".
            run_number: Run number for extracting run-specific entries. Default is None and uses generic entries only.
        """
        if plane_name == "default":
            filter_condition = lambda entry: entry['valid_begin'] == 0 and entry['valid_end'] == 0
        elif plane_name == "user":
            filter_condition = lambda entry: entry['valid_begin'] == -1 and entry['valid_end'] == -1
        elif plane_name == "run":
            if run_number is None:
                filter_condition = lambda entry: entry['valid_begin'] > 0 or entry['valid_end'] > 0
            else:
                filter_condition = lambda entry: entry['valid_begin'] <= run_number or entry['valid_end'] >= run_number
        else:
            raise ValueError(f"Invalid plane name: {plane_name}")

        entries_iter = filter(filter_condition, self.entries)
        return [dict(entry) for entry in entries_iter]

    @staticmethod
    def _merge_entry(base_entry: dict, new_entry: dict, override_base=False):
        assert base_entry['name'] == new_entry['name'], \
            f'Requesting a merge of entries with different table names! {base_entry["name"]} <- {new_entry["name"]}'
        assert base_entry['index'] == new_entry['index'], \
            f'Requesting a merge of entries with different index! {base_entry["index"]} <- {new_entry["index"]}'
        if override_base:
            result = base_entry
        else:
            result = deepcopy(base_entry)
        for key, value in new_entry.items():
            result[key] = value
        return result

    @staticmethod
    def _merge_planes(base_plane: list[dict], new_plane: list[dict]):
        merged_plane = deepcopy(base_plane)
        for new_entry in new_plane:
            current_table = None
            for base_entry in merged_plane:
                if (base_entry.get('name') == new_entry.get('name') and
                        base_entry.get('index') == new_entry.get('index')):
                    current_table = base_entry
                    break
            if current_table is None:
                merged_plane.append(new_entry)
            else:
                RatDBParser._merge_entry(current_table, new_entry, override_base=True)
        return merged_plane

    def merge_all_planes(self):
        user_tables = self.get_entries_for_plane("user")
        default_tables = self.get_entries_for_plane("default")
        run_tables = self.get_entries_for_plane("run", run_number=self.run_number)
        merged = RatDBParser._merge_planes(default_tables, run_tables)
        merged = RatDBParser._merge_planes(merged, user_tables)
        self.entries = merged

    def create_db(self) -> dict | None:
        db = {}
        for entry in self.entries:
            name = entry.get('name')
            index = entry.get('index')
            if name not in db:
                db[name] = {}
            if index in db[name]:
                raise ValueError(f"Duplicate entry found for {name} index {index}")
            db[name][index] = entry
        return db

    def get_entry(self, table_name: str, index: str) -> dict | None:
        if self.db is None:
            logger.warn("Data base is not merged, no hash table is created since entry uniqueness is not guaranteed.")
            for entry in self.entries:
                if entry.get('name') == table_name and entry.get('index') == index:
                    return entry
            return None
        return self.db.get(table_name, {}).get(index, None)

    def get_table(self, table_name: str, as_list=False) -> dict | list | None:
        if self.db is None:
            if as_list:
                return [entry for entry in self.entries if entry.get('name') == table_name]
            logger.warn("Data base is not merged, no hash table is created since entry uniqueness is not guaranteed.")
            return {entry.get('index'): entry for entry in self.entries if entry.get('name') == table_name}
        result = self.db.get(table_name, None)
        if as_list:
            return list(result.values()) if result is not None else []
        return result

    def get_matching_entries(self, table_name_match: Optional[Callable] = None,
                             index_match: Optional[Callable] = None,
                             content_match: Optional[Callable] = None) -> list[dict]:
        if table_name_match is None:
            table_name_match = lambda x: True
        if index_match is None:
            index_match = lambda x: True
        if content_match is None:
            content_match = lambda x: True
        return [entry for entry in self.entries if table_name_match(entry['name']) and index_match(entry['index']) and content_match(entry)]

    def __str__(self):
        return f"""
        RATDB Parser: {self.ratdb_path}
        Run number: {self.run_number}
        -------------------------
        {print_json(self.entries, indent=2)}
        """


def print_json(json_obj: dict, indent=2):
    print(json.dumps(json_obj, indent=indent))
