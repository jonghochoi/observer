"""
observer/isaac/
===============
Isaac Sim / Isaac Lab integration for video recording.

Optional — only loaded when ``skip_video=False`` in the eval config.
Imports degrade to mock mode when ``omni.*`` is unavailable so the
package can still be imported on a non-Isaac host.
"""
