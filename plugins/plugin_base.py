from abc import ABC, abstractmethod
import gradio as gr
import yaml
import os
import re
import json
import html
import logging
from typing import Optional, Dict, List, Any, Tuple
from packaging import version

logger = logging.getLogger(__name__)

class PluginBase(ABC):
    """Abstract base class for all plugins."""
    
    # Class-level constant for state directory
    STATE_DIR = "plugins/state"
    
    def __init__(self, manifest: Optional[Dict[str, Any]] = None):
        """Initialize plugin with manifest injection from factory."""
        # Initialize base attributes with defaults
        self._init_base_attributes()
        
        # Skip further initialization for base class
        if self.__class__ == PluginBase:
            return
            
        # Apply manifest if provided by factory
        if manifest:
            self.manifest = manifest
            self._init_attributes_from_manifest()
        else:
            logger.error(f"No manifest provided for plugin {self.__class__.__name__}. Plugins must be created through PluginFactory.")
            raise ValueError("Manifest is required. Use PluginFactory to create plugins.")
            
        # Ensure state directory exists
        os.makedirs(self.STATE_DIR, exist_ok=True)
    
    def _init_base_attributes(self) -> None:
        """Initialize base attributes with defaults."""
        self.name: str = "Base Plugin"
        self.description: str = "Base plugin description"
        self.version: str = "1.0.0"
        self.min_webui_version: str = "1.0.0"
        self.max_webui_version: Optional[str] = None
        self.author: Optional[str] = None
        self.license: Optional[str] = None
        self.config: Dict[str, Any] = {}
        self.manifest: Dict[str, Any] = {}
        self.state: Dict[str, Any] = {}
        self.enabled: bool = True
    
    def _init_attributes_from_manifest(self) -> None:
        """Initialize attributes from manifest with defaults."""
        if not self.manifest:
            logger.error(f"Cannot initialize attributes: No manifest data for {self.__class__.__name__}")
            return
            
        self.name = self.manifest.get('name', self.name)
        self.description = self.manifest.get('description', self.description)
        self.version = self.manifest.get('version', self.version)
        self.min_webui_version = self.manifest.get('min_webui_version', self.min_webui_version)
        self.max_webui_version = self.manifest.get('max_webui_version')
        self.author = self.manifest.get('author')
        self.license = self.manifest.get('license')
        
        # Load configuration
        self.config = self.manifest.get('config', {})
        self.enabled = self.get_config('enabled', True)
        
        # Initialize state
        self.state = {}
        self._load_state()
    
    def _get_state_file_path(self) -> str:
        """Get the path to the state file for this plugin."""
        if not self.name:
            raise ValueError("Plugin name not set. Cannot determine state file path.")
        safe_name = re.sub(r'[^a-zA-Z0-9_-]', '_', self.name.lower())
        return os.path.join(self.STATE_DIR, f"{safe_name}_state.json")
    
    def _load_state(self) -> None:
        """Load persisted plugin state from dedicated state directory"""
        try:
            state_file = self._get_state_file_path()
            if os.path.exists(state_file):
                with open(state_file, 'r') as f:
                    self.state = json.load(f)
                logger.debug(f"Loaded state for plugin {self.name} from {state_file}")
            else:
                logger.debug(f"No existing state file found for plugin {self.name}")
                self.state = {}
        except Exception as e:
            logger.error(f"Failed to load state for {self.name}: {str(e)}")
            self.state = {}
    
    def _save_state(self) -> None:
        """Persist plugin state to dedicated state directory"""
        try:
            state_file = self._get_state_file_path()
            os.makedirs(os.path.dirname(state_file), exist_ok=True)
            with open(state_file, 'w') as f:
                json.dump(self.state, f, indent=2)
            logger.debug(f"Saved state for plugin {self.name} to {state_file}")
        except Exception as e:
            logger.error(f"Failed to save state for {self.name}: {str(e)}")
    
    def _sanitize_input(self, input_str: str) -> str:
        """Sanitize user input to prevent injection attacks"""
        if not isinstance(input_str, str):
            return str(input_str)
            
        # Remove any potential script tags
        input_str = re.sub(r'<script.*?>.*?</script>', '', input_str, flags=re.DOTALL)
        # Remove potential event handlers
        input_str = re.sub(r' on\w+=".*?"', '', input_str)
        # HTML escape the input
        return html.escape(input_str)
    
    def _validate_permissions(self) -> Tuple[bool, str]:
        """Validate that the plugin has required permissions"""
        required_permissions = self.manifest.get('security', {}).get('required_permissions', [])
        global_security = self.get_config('security', {})
        
        for permission in required_permissions:
            if permission == 'network_access' and not global_security.get('allow_network_access'):
                return False, f"Network access denied for {self.name}"
            elif permission == 'file_access' and not global_security.get('allow_file_access'):
                return False, f"File access denied for {self.name}"
            elif permission == 'system_access' and not global_security.get('allow_system_access'):
                return False, f"System access denied for {self.name}"
        
        return True, "All permissions granted"
    
    # Lifecycle hooks
    def on_init(self) -> None:
        """Called after plugin initialization"""
        pass
    
    def on_enable(self) -> None:
        """Called when plugin is enabled"""
        pass
    
    def on_disable(self) -> None:
        """Called when plugin is disabled"""
        pass
    
    def on_unload(self) -> None:
        """Called before plugin is unloaded"""
        self._save_state()
    
    @abstractmethod
    def is_enabled(self) -> bool:
        """Check if the plugin is enabled."""
        pass
    
    @abstractmethod
    def create_ui(self, main_tabs: gr.Tabs) -> None:
        """Create the plugin's UI components."""
        pass
    
    def get_version(self) -> str:
        """Get the plugin version."""
        return self.version
    
    def validate_compatibility(self, webui_version: str) -> Tuple[bool, str]:
        """Validate plugin compatibility with webui version."""
        try:
            current = version.parse(webui_version)
            minimum = version.parse(self.min_webui_version)
            
            if current < minimum:
                return False, f"Requires minimum version {self.min_webui_version}"
            
            if self.max_webui_version:
                maximum = version.parse(self.max_webui_version)
                if current > maximum:
                    return False, f"Requires maximum version {self.max_webui_version}"
            
            return True, "Compatible"
        except Exception as e:
            return False, f"Version validation error: {str(e)}"
    
    def get_config(self, key: str, default: Any = None) -> Any:
        """Get a configuration value."""
        return self.config.get(key, default)
    
    def set_config(self, key: str, value: Any) -> None:
        """Set a configuration value and persist it."""
        self.config[key] = value
        self.manifest['config'] = self.config
        
        # Save updated manifest
        manifest_path = os.path.join(os.path.dirname(self.__class__.__module__), "manifest.yaml")
        try:
            with open(manifest_path, 'w') as f:
                yaml.safe_dump(self.manifest, f)
        except Exception as e:
            logger.error(f"Failed to save config for {self.name}: {str(e)}")
    
    def get_state(self, key: str, default: Any = None) -> Any:
        """Get a state value."""
        return self.state.get(key, default)
    
    def set_state(self, key: str, value: Any) -> None:
        """Set a state value and persist it."""
        self.state[key] = value
        self._save_state() 