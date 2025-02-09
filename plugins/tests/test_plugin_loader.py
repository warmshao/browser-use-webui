import unittest
import os
import tempfile
import shutil
import yaml
import subprocess
from typing import Dict, Any
from datetime import datetime
from plugins import load_plugins, PluginBase, plugin_manager, PluginFactory

class TestPluginSystem(unittest.TestCase):
    def setUp(self):
        """Set up test environment."""
        # Create a temporary directory for test plugins
        self.test_dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.test_dir)
        
        # Create plugins directory
        self.plugins_dir = os.path.join(self.test_dir, "plugins")
        os.makedirs(self.plugins_dir)
        
        # Create config file with proper security settings
        self.config_path = os.path.join(self.plugins_dir, "config.yaml")
        with open(self.config_path, 'w') as f:
            yaml.dump({
                "enabled_plugins": [],
                "plugin_config": {},
                "security": {
                    "allow_network_access": True,  # Explicitly allow network access
                    "allow_file_access": True,
                    "allow_system_access": True,
                    "sanitize_all_inputs": True
                }
            }, f)
    
    def create_test_plugin(self, name: str, manifest_data: Dict[str, Any], plugin_content: str) -> None:
        """Helper to create a test plugin with manifest."""
        plugin_dir = os.path.join(self.plugins_dir, name)
        os.makedirs(plugin_dir)
        
        # Create plugin file
        with open(os.path.join(plugin_dir, f"{name}_plugin.py"), 'w') as f:
            f.write(plugin_content)
            
        # Create manifest file
        manifest_data['config'] = manifest_data.get('config', {})
        manifest_data['config']['security'] = {  # Add security config to manifest
            'allow_network_access': True,
            'allow_file_access': True,
            'allow_system_access': True
        }
        with open(os.path.join(plugin_dir, "manifest.yaml"), 'w') as f:
            yaml.dump(manifest_data, f)
            
        # Create empty __init__.py
        with open(os.path.join(plugin_dir, "__init__.py"), 'w') as f:
            f.write("")
        
        # Update config to enable the plugin
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
        config["enabled_plugins"].append(name)
        with open(self.config_path, 'w') as f:
            yaml.dump(config, f)
    
    def _get_package_version(self, package: str) -> str:
        """Helper to get package version."""
        result = subprocess.run(
            ['pip', 'show', package],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            for line in result.stdout.split('\n'):
                if line.startswith('Version: '):
                    return line.split('Version: ')[1].strip()
        return "0.0.0"  # Return minimal version if not found

    def test_factory_create_plugin(self):
        """Test plugin creation using factory."""
        manifest_data = {
            "name": "Test Plugin",
            "version": "1.0.0",
            "description": "A test plugin",
            "min_webui_version": "1.0.0",
            "config": {"enabled": True}
        }
        
        plugin_content = '''
from plugins.plugin_base import PluginBase
import gradio as gr
from typing import Optional, Dict, Any

class TestPlugin(PluginBase):
    def __init__(self, manifest: Optional[Dict[str, Any]] = None):
        super().__init__(manifest=manifest)
        
    def is_enabled(self) -> bool:
        has_permissions, _ = self._validate_permissions()
        return self.enabled and has_permissions
        
    def create_ui(self, main_tabs: gr.Tabs) -> None:
        pass
'''
        self.create_test_plugin("test", manifest_data, plugin_content)
        
        # Test factory creation
        plugin = PluginFactory.create_plugin(self.plugins_dir, "test")
        self.assertIsNotNone(plugin, "Plugin should be created successfully")
        if plugin:  # Add null check
            self.assertEqual(plugin.name, "Test Plugin")
            self.assertEqual(plugin.version, "1.0.0")
    
    def test_dependency_injection(self):
        """Test manifest injection into plugin."""
        manifest = {
            "name": "Injected Plugin",
            "version": "2.0.0",
            "description": "Testing injection",
            "min_webui_version": "1.0.0",
            "config": {"custom_setting": "test_value"}
        }
        
        plugin_content = '''
from plugins.plugin_base import PluginBase
import gradio as gr
from typing import Optional, Dict, Any

class TestPlugin(PluginBase):
    def __init__(self, manifest: Optional[Dict[str, Any]] = None):
        super().__init__(manifest=manifest)
        
    def is_enabled(self) -> bool:
        return True
        
    def create_ui(self, main_tabs: gr.Tabs) -> None:
        pass
'''
        self.create_test_plugin("test_inject", manifest, plugin_content)
        
        # Test plugin creation with injected manifest
        plugin = PluginFactory.create_plugin(self.plugins_dir, "test_inject")
        self.assertIsNotNone(plugin, "Plugin should be created successfully")
        if plugin:  # Add null check
            self.assertEqual(plugin.name, "Injected Plugin")
            self.assertEqual(plugin.get_config("custom_setting"), "test_value")
    
    def test_plugin_lifecycle(self):
        """Test plugin lifecycle hooks."""
        manifest = {
            "name": "Lifecycle Plugin",
            "version": "1.0.0",
            "description": "Testing lifecycle",
            "min_webui_version": "1.0.0",
            "config": {"enabled": True}
        }
        
        plugin_content = '''
from plugins.plugin_base import PluginBase
import gradio as gr
from typing import Optional, Dict, Any
from datetime import datetime

class TestPlugin(PluginBase):
    def __init__(self, manifest: Optional[Dict[str, Any]] = None):
        super().__init__(manifest=manifest)
        self.set_state('init_time', datetime.now().isoformat())
        
    def is_enabled(self) -> bool:
        return True
        
    def create_ui(self, main_tabs: gr.Tabs) -> None:
        pass
        
    def on_enable(self) -> None:
        self.set_state('enabled_time', datetime.now().isoformat())
        
    def on_disable(self) -> None:
        self.set_state('disabled_time', datetime.now().isoformat())
'''
        self.create_test_plugin("test_lifecycle", manifest, plugin_content)
        
        # Test lifecycle
        plugin = PluginFactory.create_plugin(self.plugins_dir, "test_lifecycle")
        self.assertIsNotNone(plugin, "Plugin should be created successfully")
        if plugin:  # Add null check
            self.assertIsNotNone(plugin.get_state('init_time'))
            
            # Test enable/disable
            plugin.on_enable()
            self.assertIsNotNone(plugin.get_state('enabled_time'))
            
            plugin.on_disable()
            self.assertIsNotNone(plugin.get_state('disabled_time'))
    
    def test_plugin_security(self):
        """Test plugin security features."""
        manifest = {
            "name": "Security Plugin",
            "version": "1.0.0",
            "description": "Testing security",
            "min_webui_version": "1.0.0",
            "config": {"enabled": True},
            "security": {
                "required_permissions": ["network_access"],
                "input_sanitization": ["test_input"]
            }
        }
        
        plugin_content = '''
from plugins.plugin_base import PluginBase
import gradio as gr
from typing import Optional, Dict, Any

class TestPlugin(PluginBase):
    def __init__(self, manifest: Optional[Dict[str, Any]] = None):
        super().__init__(manifest=manifest)
        
    def is_enabled(self) -> bool:
        has_permissions, _ = self._validate_permissions()
        return self.enabled and has_permissions
        
    def create_ui(self, main_tabs: gr.Tabs) -> None:
        pass
        
    def sanitize_test_input(self, text: str) -> str:
        """Public method for testing input sanitization."""
        return self._sanitize_input(text)
'''
        self.create_test_plugin("test_security", manifest, plugin_content)
        
        # Test security features
        plugin = PluginFactory.create_plugin(self.plugins_dir, "test_security")
        self.assertIsNotNone(plugin, "Plugin should be created successfully")
        if plugin:  # Add null check
            # Test permissions
            has_permissions, _ = plugin._validate_permissions()
            self.assertTrue(has_permissions)
            
            # Test input sanitization
            test_input = '<script>alert("xss")</script>Hello'
            sanitized = plugin.sanitize_test_input(test_input)
            self.assertNotIn('<script>', sanitized)
            self.assertIn('Hello', sanitized)
    
    def test_version_compatibility(self):
        """Test version compatibility checking."""
        manifest = {
            "name": "Version Plugin",
            "version": "1.0.0",
            "description": "Testing versions",
            "min_webui_version": "2.0.0",  # Incompatible
            "max_webui_version": "3.0.0",
            "config": {"enabled": True}
        }
        
        plugin_content = '''
from plugins.plugin_base import PluginBase
import gradio as gr
from typing import Optional, Dict, Any

class TestPlugin(PluginBase):
    def __init__(self, manifest: Optional[Dict[str, Any]] = None):
        super().__init__(manifest=manifest)
        
    def is_enabled(self) -> bool:
        return True
        
    def create_ui(self, main_tabs: gr.Tabs) -> None:
        pass
'''
        self.create_test_plugin("test_version", manifest, plugin_content)
        
        # Test version compatibility
        plugin = PluginFactory.create_plugin(self.plugins_dir, "test_version")
        self.assertIsNotNone(plugin, "Plugin should be created successfully")
        if plugin:  # Add null check
            is_compatible, reason = plugin.validate_compatibility("1.0.0")
            self.assertFalse(is_compatible)
            self.assertIn("minimum version", reason)
    
    def test_plugin_state_management(self):
        """Test plugin state file management."""
        manifest = {
            "name": "State Test Plugin",
            "version": "1.0.0",
            "description": "Testing state management",
            "min_webui_version": "1.0.0",
            "config": {"enabled": True}
        }
        
        plugin_content = '''
from plugins.plugin_base import PluginBase
import gradio as gr
from typing import Optional, Dict, Any

class TestPlugin(PluginBase):
    def __init__(self, manifest: Optional[Dict[str, Any]] = None):
        super().__init__(manifest=manifest)
        
    def is_enabled(self) -> bool:
        return True
        
    def create_ui(self, main_tabs: gr.Tabs) -> None:
        pass
'''
        self.create_test_plugin("test_state", manifest, plugin_content)
        
        # Create plugin instance
        plugin = PluginFactory.create_plugin(self.plugins_dir, "test_state")
        self.assertIsNotNone(plugin, "Plugin should be created successfully")
        if plugin:
            # Test state file location
            state_file = plugin._get_state_file_path()
            self.assertTrue(state_file.endswith("state_test_plugin_state.json"))
            self.assertTrue(state_file.startswith(plugin.STATE_DIR))
            
            # Test state persistence
            test_state = {"test_key": "test_value"}
            plugin.state = test_state
            plugin._save_state()
            
            # Verify file exists
            self.assertTrue(os.path.exists(state_file))
            
            # Create new instance and verify state loads
            plugin2 = PluginFactory.create_plugin(self.plugins_dir, "test_state")
            self.assertIsNotNone(plugin2)
            if plugin2:
                self.assertEqual(plugin2.state.get("test_key"), "test_value")
                
            # Test state file cleanup
            os.remove(state_file)
            plugin3 = PluginFactory.create_plugin(self.plugins_dir, "test_state")
            self.assertIsNotNone(plugin3)
            if plugin3:
                self.assertEqual(plugin3.state, {})  # Empty state for new instance

    def test_dependency_management(self):
        """Test plugin dependency management."""
        manifest = {
            "name": "Dependency Plugin",
            "version": "1.0.0",
            "description": "Testing dependencies",
            "min_webui_version": "1.0.0",
            "config": {
                "enabled": True,
                "auto_install_deps": True
            },
            "dependencies": [
                {"package": "gradio>=3.50.0"},
                {"package": "pyyaml>=6.0.1"}
            ]
        }
        
        plugin_content = '''
from plugins.plugin_base import PluginBase
import gradio as gr
from typing import Optional, Dict, Any

class TestPlugin(PluginBase):
    def __init__(self, manifest: Optional[Dict[str, Any]] = None):
        super().__init__(manifest=manifest)
        
    def is_enabled(self) -> bool:
        return True
        
    def create_ui(self, main_tabs: gr.Tabs) -> None:
        pass
'''
        self.create_test_plugin("test_deps", manifest, plugin_content)
        
        # Test dependency installation
        plugin = PluginFactory.create_plugin(self.plugins_dir, "test_deps")
        self.assertIsNotNone(plugin, "Plugin should be created successfully")
        
        # Verify gradio is installed with correct version
        gradio_version = self._get_package_version("gradio")
        self.assertTrue(PluginFactory._check_version(gradio_version, "3.50.0"))
        
        # Verify pyyaml is installed with correct version
        yaml_version = self._get_package_version("pyyaml")
        self.assertTrue(PluginFactory._check_version(yaml_version, "6.0.1"))
        
    def test_dependency_skip_when_satisfied(self):
        """Test that dependencies aren't reinstalled if already satisfied."""
        current_gradio = self._get_package_version("gradio")
        manifest = {
            "name": "Skip Deps Plugin",
            "version": "1.0.0",
            "description": "Testing dependency skipping",
            "min_webui_version": "1.0.0",
            "config": {
                "enabled": True,
                "auto_install_deps": True
            },
            "dependencies": [
                {"package": f"gradio>={current_gradio}"}
            ]
        }
        
        plugin_content = '''
from plugins.plugin_base import PluginBase
import gradio as gr
from typing import Optional, Dict, Any

class TestPlugin(PluginBase):
    def __init__(self, manifest: Optional[Dict[str, Any]] = None):
        super().__init__(manifest=manifest)
        
    def is_enabled(self) -> bool:
        return True
        
    def create_ui(self, main_tabs: gr.Tabs) -> None:
        pass
'''
        self.create_test_plugin("test_skip_deps", manifest, plugin_content)
        
        # Test dependency skipping
        plugin = PluginFactory.create_plugin(self.plugins_dir, "test_skip_deps")
        self.assertIsNotNone(plugin, "Plugin should be created successfully")
        
    def test_dependency_disabled(self):
        """Test that dependencies aren't installed when auto_install_deps is disabled."""
        manifest = {
            "name": "No Deps Plugin",
            "version": "1.0.0",
            "description": "Testing disabled dependencies",
            "min_webui_version": "1.0.0",
            "config": {
                "enabled": True,
                "auto_install_deps": False
            },
            "dependencies": [
                {"package": "nonexistent-package>=1.0.0"}
            ]
        }
        
        plugin_content = '''
from plugins.plugin_base import PluginBase
import gradio as gr
from typing import Optional, Dict, Any

class TestPlugin(PluginBase):
    def __init__(self, manifest: Optional[Dict[str, Any]] = None):
        super().__init__(manifest=manifest)
        
    def is_enabled(self) -> bool:
        return True
        
    def create_ui(self, main_tabs: gr.Tabs) -> None:
        pass
'''
        self.create_test_plugin("test_no_deps", manifest, plugin_content)
        
        # Test dependency installation is skipped
        plugin = PluginFactory.create_plugin(self.plugins_dir, "test_no_deps")
        self.assertIsNotNone(plugin, "Plugin should be created successfully")
        
        # Verify nonexistent package wasn't installed
        result = subprocess.run(['pip', 'show', 'nonexistent-package'], capture_output=True)
        self.assertNotEqual(result.returncode, 0, "Package should not be installed")

    def test_invalid_dependency_format(self):
        """Test handling of invalid dependency specifications."""
        manifest = {
            "name": "Invalid Deps Plugin",
            "version": "1.0.0",
            "description": "Testing invalid dependencies",
            "min_webui_version": "1.0.0",
            "config": {
                "enabled": True,
                "auto_install_deps": True
            },
            "dependencies": [
                {"wrong_key": "gradio>=3.50.0"},
                {},  # Empty dependency
                {"package": ""}  # Empty package name
            ]
        }
        
        plugin_content = '''
from plugins.plugin_base import PluginBase
import gradio as gr
from typing import Optional, Dict, Any

class TestPlugin(PluginBase):
    def __init__(self, manifest: Optional[Dict[str, Any]] = None):
        super().__init__(manifest=manifest)
        
    def is_enabled(self) -> bool:
        return True
        
    def create_ui(self, main_tabs: gr.Tabs) -> None:
        pass
'''
        self.create_test_plugin("test_invalid_deps", manifest, plugin_content)
        
        # Test invalid dependencies are handled gracefully
        plugin = PluginFactory.create_plugin(self.plugins_dir, "test_invalid_deps")
        self.assertIsNotNone(plugin, "Plugin should be created despite invalid dependencies")

if __name__ == '__main__':
    unittest.main() 