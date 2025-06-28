#!/usr/bin/env python3
"""
Enhanced Flask Documentation Scanner
Scans blueprint folders, services, and biomechanic utilities to generate comprehensive documentation
"""

import ast
import os
import re
from pathlib import Path
from typing import Dict, List, Any, Optional
import json
import logging
import inspect

logger = logging.getLogger(__name__)

class FlaskDocumentationScanner:
    def __init__(self, blueprint_folder: str, services_folder: str = "services", 
                 runnervision_folder: str = "runnervision_utils"):
        self.blueprint_folder = Path(blueprint_folder)
        self.services_folder = Path(services_folder)
        self.runnervision_folder = Path(runnervision_folder)
        self.documentation = {
            'blueprints': {},
            'routes': [],
            'services': {},
            'biomechanic_modules': {},
            'summary': {}
        }
    
    def scan_all_blueprints(self):
        """Scan all blueprint directories and generate documentation"""
        for bp_dir in self.blueprint_folder.iterdir():
            if bp_dir.is_dir() and not bp_dir.name.startswith('.'):
                self.scan_blueprint_directory(bp_dir)
        
        self.generate_summary()
        return self.documentation
    
    def scan_services(self):
        """Scan services directory for service classes and functions"""
        if not self.services_folder.exists():
            return
        
        for service_file in self.services_folder.glob("*.py"):
            if service_file.name.startswith('__'):
                continue
                
            service_info = self.analyze_service_file(service_file)
            if service_info:
                self.documentation['services'][service_file.stem] = service_info
    
    def scan_runnervision_utils(self):
        """Scan runnervision_utils directory for biomechanic modules"""
        if not self.runnervision_folder.exists():
            return
        
        for py_file in self.runnervision_folder.rglob("*.py"):
            if py_file.name.startswith('__'):
                continue
                
            module_info = self.analyze_biomechanic_module(py_file)
            if module_info:
                relative_path = py_file.relative_to(self.runnervision_folder)
                module_key = str(relative_path.with_suffix(''))
                self.documentation['biomechanic_modules'][module_key] = module_info
    
    def analyze_service_file(self, service_file: Path) -> Optional[Dict[str, Any]]:
        """Analyze a service file for classes and functions"""
        try:
            with open(service_file, 'r', encoding='utf-8') as f:
                content = f.read()
                tree = ast.parse(content)
        except Exception as e:
            print(f"Error parsing {service_file}: {e}")
            return None
        
        service_info = {
            'file_path': str(service_file),
            'classes': [],
            'functions': [],
            'imports': self.extract_imports(tree)
        }
        
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                class_info = self.analyze_class(node)
                service_info['classes'].append(class_info)
            elif isinstance(node, ast.FunctionDef) and not self.is_method(node, tree):
                func_info = self.analyze_function(node)
                service_info['functions'].append(func_info)
        
        return service_info if service_info['classes'] or service_info['functions'] else None
    
    def analyze_biomechanic_module(self, module_file: Path) -> Optional[Dict[str, Any]]:
        """Analyze a biomechanic module file"""
        try:
            with open(module_file, 'r', encoding='utf-8') as f:
                content = f.read()
                tree = ast.parse(content)
        except Exception as e:
            print(f"Error parsing {module_file}: {e}")
            return None
        
        module_info = {
            'file_path': str(module_file),
            'module_docstring': ast.get_docstring(tree),
            'classes': [],
            'functions': [],
            'imports': self.extract_imports(tree)
        }
        
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                class_info = self.analyze_class(node)
                module_info['classes'].append(class_info)
            elif isinstance(node, ast.FunctionDef) and not self.is_method(node, tree):
                func_info = self.analyze_function(node)
                module_info['functions'].append(func_info)
        
        return module_info if (module_info['classes'] or module_info['functions'] or 
                             module_info['module_docstring']) else None
    
    def analyze_class(self, class_node: ast.ClassDef) -> Dict[str, Any]:
        """Analyze a class definition"""
        class_info = {
            'name': class_node.name,
            'docstring': ast.get_docstring(class_node),
            'methods': [],
            'line_number': class_node.lineno,
            'bases': [self.get_node_name(base) for base in class_node.bases]
        }
        
        for node in class_node.body:
            if isinstance(node, ast.FunctionDef):
                method_info = self.analyze_function(node, is_method=True)
                class_info['methods'].append(method_info)
        
        return class_info
    
    def analyze_function(self, func_node: ast.FunctionDef, is_method: bool = False) -> Dict[str, Any]:
        """Analyze a function definition"""
        # Extract parameters, excluding 'self' for methods
        params = []
        for arg in func_node.args.args:
            if not (is_method and arg.arg == 'self'):
                param_info = {'name': arg.arg}
                if arg.annotation:
                    param_info['type'] = self.get_node_name(arg.annotation)
                params.append(param_info)
        
        # Extract return type annotation
        return_type = None
        if func_node.returns:
            return_type = self.get_node_name(func_node.returns)
        
        func_info = {
            'name': func_node.name,
            'docstring': ast.get_docstring(func_node),
            'parameters': params,
            'return_type': return_type,
            'line_number': func_node.lineno,
            'is_async': isinstance(func_node, ast.AsyncFunctionDef),
            'decorators': [self.get_node_name(dec) for dec in func_node.decorator_list]
        }
        
        return func_info
    
    def get_node_name(self, node: ast.AST) -> str:
        """Extract name from various AST node types"""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return f"{self.get_node_name(node.value)}.{node.attr}"
        elif isinstance(node, ast.Constant):
            return str(node.value)
        elif isinstance(node, ast.Subscript):
            return f"{self.get_node_name(node.value)}[{self.get_node_name(node.slice)}]"
        else:
            return ast.unparse(node) if hasattr(ast, 'unparse') else str(node)
    
    def is_method(self, func_node: ast.FunctionDef, tree: ast.AST) -> bool:
        """Check if a function is a method inside a class"""
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                if func_node in node.body:
                    return True
        return False
    
    def scan_blueprint_directory(self, bp_dir: Path):
        """Scan a single blueprint directory"""
        bp_name = bp_dir.name
        routes_file = bp_dir / 'routes.py'
        
        if not routes_file.exists():
            return
        
        blueprint_info = {
            'name': bp_name,
            'directory': str(bp_dir),
            'url_prefix': None,
            'blueprint_variable': None,
            'routes': [],
            'imports': [],
            'services_used': []
        }
        
        # Parse the routes.py file
        with open(routes_file, 'r', encoding='utf-8') as f:
            content = f.read()
            tree = ast.parse(content)
        
        # Extract blueprint definition and routes
        blueprint_info.update(self.extract_blueprint_info(tree, content))
        blueprint_info['routes'] = self.extract_routes(tree, blueprint_info['blueprint_variable'])
        blueprint_info['imports'] = self.extract_imports(tree)
        blueprint_info['services_used'] = self.extract_service_usage(content)
        
        self.documentation['blueprints'][bp_name] = blueprint_info
        
        # Add routes to global routes list
        for route in blueprint_info['routes']:
            route['blueprint'] = bp_name
            route['full_url'] = f"{blueprint_info['url_prefix'] or ''}{route['endpoint']}"
            self.documentation['routes'].append(route)
    
    def extract_blueprint_info(self, tree: ast.AST, content: str) -> Dict[str, Any]:
        """Extract blueprint definition information"""
        info = {}
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                # Look for Blueprint assignments
                for target in node.targets:
                    if isinstance(target, ast.Name) and isinstance(node.value, ast.Call):
                        if (hasattr(node.value.func, 'id') and 
                            node.value.func.id == 'Blueprint'):
                            info['blueprint_variable'] = target.id
                            
                            # Extract url_prefix from Blueprint arguments
                            for keyword in node.value.keywords:
                                if keyword.arg == 'url_prefix':
                                    if isinstance(keyword.value, ast.Constant):
                                        info['url_prefix'] = keyword.value.value
        
        return info
    
    def extract_routes(self, tree: ast.AST, blueprint_var: str) -> List[Dict[str, Any]]:
        """Extract route information from the AST"""
        routes = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                route_info = self.analyze_route_function(node, blueprint_var)
                if route_info:
                    routes.append(route_info)
        
        return routes
    
    def analyze_route_function(self, func_node: ast.FunctionDef, blueprint_var: str) -> Dict[str, Any]:
        """Analyze a function to extract route information"""
        route_decorators = []
        
        for decorator in func_node.decorator_list:
            if isinstance(decorator, ast.Call):
                # Handle @blueprint.route() style decorators
                if (isinstance(decorator.func, ast.Attribute) and
                    isinstance(decorator.func.value, ast.Name) and
                    decorator.func.value.id == blueprint_var and
                    decorator.func.attr == 'route'):
                    
                    route_info = {
                        'function_name': func_node.name,
                        'endpoint': None,
                        'methods': ['GET'],  # Default
                        'docstring': ast.get_docstring(func_node),
                        'parameters': [arg.arg for arg in func_node.args.args if arg.arg != 'self'],
                        'line_number': func_node.lineno
                    }
                    
                    # Extract route path
                    if decorator.args:
                        if isinstance(decorator.args[0], ast.Constant):
                            route_info['endpoint'] = decorator.args[0].value
                    
                    # Extract methods
                    for keyword in decorator.keywords:
                        if keyword.arg == 'methods':
                            if isinstance(keyword.value, ast.List):
                                methods = []
                                for method in keyword.value.elts:
                                    if isinstance(method, ast.Constant):
                                        methods.append(method.value)
                                route_info['methods'] = methods
                    
                    return route_info
        
        return None
    
    def extract_imports(self, tree: ast.AST) -> List[str]:
        """Extract import statements"""
        imports = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ''
                for alias in node.names:
                    imports.append(f"{module}.{alias.name}" if module else alias.name)
        
        return imports
    
    def extract_service_usage(self, content: str) -> List[str]:
        """Extract service class usage from content"""
        services = []
        
        # Look for common service patterns
        service_patterns = [
            r'(\w*Service)\(\)',
            r'(\w*service)\.',
            r'from.*services.*import\s+(\w+)',
        ]
        
        for pattern in service_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            services.extend(matches)
        
        return list(set(services))  # Remove duplicates
    
    def generate_summary(self):
        """Generate summary statistics"""
        total_routes = len(self.documentation['routes'])
        total_blueprints = len(self.documentation['blueprints'])
        total_services = len(self.documentation['services'])
        total_biomechanic_modules = len(self.documentation['biomechanic_modules'])
        
        methods_count = {}
        for route in self.documentation['routes']:
            for method in route['methods']:
                methods_count[method] = methods_count.get(method, 0) + 1
        
        self.documentation['summary'] = {
            'total_blueprints': total_blueprints,
            'total_routes': total_routes,
            'total_services': total_services,
            'total_biomechanic_modules': total_biomechanic_modules,
            'methods_distribution': methods_count,
            'blueprints_list': list(self.documentation['blueprints'].keys()),
            'services_list': list(self.documentation['services'].keys()),
            'biomechanic_modules_list': list(self.documentation['biomechanic_modules'].keys())
        }
    
    def generate_markdown_report(self) -> str:
        """Generate a markdown documentation report"""
        md = "# Flask Application Documentation  \n"
        md += "  \n"
        
        # Summary
        summary = self.documentation['summary']
        md += "## Summary  \n"
        md += "  \n"
        md += f"- **Total Blueprints**: {summary['total_blueprints']}  \n"
        md += f"- **Total Routes**: {summary['total_routes']}  \n"
        md += f"- **Total Services**: {summary['total_services']}  \n"
        md += f"- **Total Biomechanic Modules**: {summary['total_biomechanic_modules']}  \n"
        md += f"- **HTTP Methods**: {', '.join(f'{k}({v})' for k, v in summary['methods_distribution'].items())}  \n"
        md += "  \n"
        
        # Blueprint Details
        for bp_name, bp_info in self.documentation['blueprints'].items():
            md += f"## Blueprint: {bp_name}  \n"
            md += "  \n"
            md += f"**URL Prefix**: `{bp_info['url_prefix'] or 'None'}`  \n"
            md += f"**Directory**: `{bp_info['directory']}`  \n"
            md += f"**Routes Count**: {len(bp_info['routes'])}  \n"
            md += "  \n"
            
            if bp_info['services_used']:
                md += f"**Services Used**: {', '.join(bp_info['services_used'])}  \n"
                md += "  \n"
            
            md += "### Routes  \n"
            md += "  \n"
            for route in bp_info['routes']:
                methods_str = ', '.join(route['methods'])
                full_url = f"{bp_info['url_prefix'] or ''}{route['endpoint']}"
                md += f"- **{methods_str}** `{full_url}` → `{route['function_name']}()`  \n"
                
                if route['docstring']:
                    md += f"    - {route['docstring'].split('.')[0]}  \n"
                
                if route['parameters']:
                    params = [p for p in route['parameters'] if p not in ['request', 'session']]
                    if params:
                        md += f"    - Parameters: {', '.join(params)}  \n"
            
            md += "  \n"
        
        return md
    
    def generate_services_markdown(self) -> str:
        """Generate services documentation in markdown format"""
        md = "# Services Documentation  \n"
        md += "  \n"
        
        if not self.documentation['services']:
            md += "No services found.  \n"
            return md
        
        md += f"Found {len(self.documentation['services'])} service files.  \n"
        md += "  \n"
        
        for service_name, service_info in self.documentation['services'].items():
            md += f"## Service: {service_name}  \n"
            md += "  \n"
            md += f"**File**: `{service_info['file_path']}`  \n"
            md += "  \n"
            
            # Classes
            if service_info['classes']:
                md += "### Classes  \n"
                md += "  \n"
                for class_info in service_info['classes']:
                    md += f"#### {class_info['name']}  \n"
                    md += "  \n"
                    if class_info['docstring']:
                        md += f"{class_info['docstring']}  \n"
                        md += "  \n"
                    
                    if class_info['bases']:
                        md += f"**Inherits from**: {', '.join(class_info['bases'])}  \n"
                        md += "  \n"
                    
                    if class_info['methods']:
                        md += "**Methods**:  \n"
                        md += "  \n"
                        for method in class_info['methods']:
                            params = ', '.join([p['name'] for p in method['parameters']])
                            md += f"- `{method['name']}({params})`"
                            if method['return_type']:
                                md += f" → {method['return_type']}"
                            md += "  \n"
                            if method['docstring']:
                                # Get first line of docstring
                                first_line = method['docstring'].split('\n')[0].strip()
                                md += f"    - {first_line}  \n"
                        md += "  \n"
            
            # Standalone functions
            if service_info['functions']:
                md += "### Functions  \n"
                md += "  \n"
                for func in service_info['functions']:
                    params = ', '.join([p['name'] for p in func['parameters']])
                    md += f"#### {func['name']}({params})"
                    if func['return_type']:
                        md += f" → {func['return_type']}"
                    md += "  \n"
                    md += "  \n"
                    if func['docstring']:
                        md += f"{func['docstring']}  \n"
                        md += "  \n"
            
            md += "---  \n"
            md += "  \n"
        
        return md
    
    def generate_biomechanic_markdown(self) -> str:
        """Generate biomechanic modules documentation in markdown format"""
        md = "# RunnerVision Biomechanic Documentation  \n"
        md += "  \n"
        
        if not self.documentation['biomechanic_modules']:
            md += "No biomechanic modules found.  \n"
            return md
        
        md += f"Found {len(self.documentation['biomechanic_modules'])} biomechanic modules.  \n"
        md += "  \n"
        
        # Group by directory for better organization
        modules_by_dir = {}
        for module_path, module_info in self.documentation['biomechanic_modules'].items():
            dir_path = str(Path(module_path).parent) if '/' in module_path else '.'
            if dir_path not in modules_by_dir:
                modules_by_dir[dir_path] = []
            modules_by_dir[dir_path].append((module_path, module_info))
        
        for dir_path, modules in modules_by_dir.items():
            if dir_path != '.':
                md += f"## Directory: {dir_path}  \n"
                md += "  \n"
            
            for module_path, module_info in modules:
                module_name = Path(module_path).name
                md += f"### Module: {module_name}  \n"
                md += f"**File**: `{module_info['file_path']}`  \n"
                md += "  \n"
                
                if module_info['module_docstring']:
                    md += f"{module_info['module_docstring']}  \n"
                    md += "  \n"
                
                # Classes
                if module_info['classes']:
                    md += "#### Classes  \n"
                    for class_info in module_info['classes']:
                        md += f"##### {class_info['name']}  \n"
                        if class_info['docstring']:
                            md += f"{class_info['docstring']}  \n"
                            md += "  \n"
                        
                        if class_info['methods']:
                            md += "**Methods**:  \n"
                            for method in class_info['methods']:
                                params = ', '.join([p['name'] for p in method['parameters']])
                                md += f"- `{method['name']}({params})`"
                                if method['return_type']:
                                    md += f" -> {method['return_type']}"
                                md += "  \n"
                                if method['docstring']:
                                    first_line = method['docstring'].split('\n')[0].strip()
                                    md += f"  - {first_line}  \n"
                            md += "  \n"
                
                # Functions
                if module_info['functions']:
                    md += "#### Functions  \n"
                    for func in module_info['functions']:
                        params = ', '.join([p['name'] for p in func['parameters']])
                        md += f"##### {func['name']}({params})"
                        if func['return_type']:
                            md += f" -> {func['return_type']}"
                        md += "  \n"
                        if func['docstring']:
                            md += f"{func['docstring']}  \n"
                        md += "  \n"
                
                md += "---  \n"
                md += "  \n"
        
        return md
    
    def save_documentation(self, output_dir: str = "docs"):
        """Save documentation in multiple formats"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Save JSON
        with open(output_path / "flask_api_documentation.json", 'w') as f:
            json.dump(self.documentation, f, indent=2)
        
        # Save main Markdown
        markdown_content = self.generate_markdown_report()
        with open(output_path / "flask_api_documentation.md", 'w', encoding='utf-8') as f:
            f.write(markdown_content)
        
        # Save Services Markdown
        services_markdown = self.generate_services_markdown()
        with open(output_path / "flask_services_documentation.md", 'w', encoding='utf-8') as f:
            f.write(services_markdown)
        
        # Save Biomechanic Markdown
        biomechanic_markdown = self.generate_biomechanic_markdown()
        with open(output_path / "runnervision_biomechanic_documentation.md", 'w', encoding='utf-8') as f:
            f.write(biomechanic_markdown)
        
        print(f"Documentation saved to {output_path}/")
        return output_path


def main():
    """Main execution function"""
    # Configure your folder paths here
    BLUEPRINT_FOLDER = "blueprints"  # Adjust this path
    SERVICES_FOLDER = "services"     # Adjust this path
    RUNNERVISION_FOLDER = "runnervision_utils"  # Adjust this path
    
    # Check if main folders exist
    missing_folders = []
    for folder, name in [(BLUEPRINT_FOLDER, "Blueprint"), (SERVICES_FOLDER, "Services"), 
                        (RUNNERVISION_FOLDER, "RunnerVision")]:
        if not Path(folder).exists():
            missing_folders.append(f"{name} folder '{folder}'")
    
    if missing_folders:
        print(f"Warning: {', '.join(missing_folders)} not found!")
    
    scanner = FlaskDocumentationScanner(BLUEPRINT_FOLDER, SERVICES_FOLDER, RUNNERVISION_FOLDER)
    
    # Scan all components
    documentation = scanner.scan_all_blueprints()
    scanner.scan_services()
    scanner.scan_runnervision_utils()
    scanner.generate_summary()  # Regenerate summary with all data
    
    # Print summary
    logger.info("Flask Documentation Scan Complete!")
    summary = scanner.documentation['summary']
    logger.info(f"Found {summary['total_blueprints']} blueprints")
    logger.info(f"Found {summary['total_routes']} routes")
    logger.info(f"Found {summary['total_services']} services")
    logger.info(f"Found {summary['total_biomechanic_modules']} biomechanic modules")
    
    # Save documentation
    output_path = scanner.save_documentation()
    
    logger.info(f"\nGenerated files:")
    logger.info(f"- flask_api_documentation.md")
    logger.info(f"- flask_services_documentation.md")
    logger.info(f"- runnervision_biomechanic_documentation.md")
    logger.info(f"- flask_api_documentation.json")


if __name__ == "__main__":
    main()