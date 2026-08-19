"""Persistent storage package for transcripts and datasets."""

from .storage_manager import (
    save_transcript_session,
    save_recognition_dataset,
    save_detection_dataset,
    list_entries,
    get_transcript_detail,
    get_dataset_detail,
    delete_entry,
    zip_entry,
    resolve_storage_root,
    ensure_storage_layout,
)

__all__ = [
    "save_transcript_session",
    "save_recognition_dataset",
    "save_detection_dataset",
    "list_entries",
    "get_transcript_detail",
    "get_dataset_detail",
    "delete_entry",
    "zip_entry",
    "resolve_storage_root",
    "ensure_storage_layout",
]
