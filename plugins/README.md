# Plugin System

A robust and extensible plugin system for modular functionality integration.

## Overview

The plugin system provides a standardized way to extend the application's functionality through plugins. It features dependency injection, manifest-based configuration, state management, and UI integration.

## Architecture

### Core Components

1. **Plugin Base (`PluginBase`):**
   - Abstract base class defining plugin interface
   - Manifest-based configuration
   - State persistence
   - Security validation
   - Input sanitization
   - Lifecycle hooks

2. **Plugin Factory (`PluginFactory`):**
   - Handles plugin instantiation
   - Manifest loading and validation
   - Plugin class discovery
   - Error handling

3. **Plugin Manager (`PluginManager`):**
   - Manages plugin lifecycle
   - Handles plugin loading/unloading
   - Configuration management
   - Security enforcement

### Directory Structure

```
plugins/
├── __init__.py          # Plugin manager and loader
├── plugin_base.py       # Abstract base class
├── factory.py           # Plugin factory
├── config.yaml          # Global configuration
├── twitter/            # Example plugin
│   ├── __init__.py
│   ├── twitter_plugin.py
│   ├── manifest.yaml
│   └── state.json
└── mock/               # Test plugin
    ├── __init__.py
    ├── mock_plugin.py
    ├── manifest.yaml
    └── state.json
```

## Features

- **Manifest-Based Configuration:**
  - Version compatibility
  - Plugin metadata
  - Configuration settings
  - Security requirements

- **State Management:**
  - Persistent state storage
  - JSON-based state files
  - Automatic state loading/saving

- **Security:**
  - Permission validation
  - Input sanitization
  - Access control
  - Error isolation

- **UI Integration:**
  - Gradio-based UI components
  - Tab-based interface
  - Dynamic UI creation
  - Error handling

- **Error Handling:**
  - Graceful failure recovery
  - Detailed error logging
  - Plugin isolation
  - Version validation

## Creating a Plugin

1. Create a new directory in `plugins/`
2. Create required files:
   ```
   your_plugin/
   ├── __init__.py
   ├── your_plugin.py
   └── manifest.yaml
   ```

3. Implement plugin class:
   ```python
   from plugins.plugin_base import PluginBase
   
   class YourPlugin(PluginBase):
       def __init__(self, manifest=None):
           super().__init__(manifest=manifest)
   
       def is_enabled(self) -> bool:
           return self.enabled
   
       def create_ui(self, main_tabs):
           # Create your UI here
           pass
   ```

4. Create manifest:
   ```yaml
   name: Your Plugin
   version: 1.0.0
   description: Your plugin description
   min_webui_version: 1.0.0
   author: Your Name
   license: MIT
   
   config:
     enabled: true
   
   security:
     sanitize_all_inputs: true
   ```

## Testing

The system includes comprehensive tests covering:
- Plugin creation and loading
- Manifest injection
- Lifecycle management
- Security validation
- Version compatibility
- UI integration
- Error handling

Run tests with:
```bash
python -m unittest plugins/tests/test_plugin_loader.py
```

## Example Plugins

1. **Twitter Plugin:**
   - Mock Twitter functionality
   - UI for posting/searching tweets
   - State persistence
   - Input validation

2. **Mock Plugin:**
   - Testing and demonstration
   - Simple text processing
   - State tracking
   - UI examples 