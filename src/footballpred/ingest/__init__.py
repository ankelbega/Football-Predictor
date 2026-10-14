"""Ingest-stage helpers for downloading, cleaning and checking match data.

The CLI invokes these modules before feature generation; source-specific quirks stay inside
ingest rather than leaking into model code.
"""
