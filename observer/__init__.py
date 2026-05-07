"""
observer/
=========
Automated evaluation pipeline for dexterous manipulation RL policies.

External consumers (training repos with their own experiment-tracking
backend) integrate via ``observer.pipeline.result_locator`` — see
``docs/22_EXTERNAL_LOGGER_HANDOFF.md``. Observer itself imports no
downstream logger; the boundary is on-disk layout plus the small
``locate_results`` / ``read_metrics`` helpers.
"""
