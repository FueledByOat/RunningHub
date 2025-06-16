#!/usr/bin/env python3
"""
Flask Blueprint Documentation Scanner
Scans blueprint folders and generates comprehensive documentation
"""

import ast
import os
import re
from pathlib import Path
from typing import Dict, List, Any
import json

class FlaskBlueprintScanner:
    def __init__(self, blueprint_folder: str):
        self.blueprint_folder = Path(blueprint_folder)
        self.documentation = {
            'blueprints': {},
            'routes': [],
            'summary': {}
        }
    
    def scan_all_blueprints(self):
        """Scan all blueprint directories and generate documentation"""
        for bp_dir in self.blueprint_folder.iterdir():
            if bp_dir.is_dir() and not bp_dir.name.startswith('.'):
                self.scan_blueprint_directory(bp_dir)
        
        self.generate_summary()
        return self.documentation
    
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
        
        methods_count = {}
        for route in self.documentation['routes']:
            for method in route['methods']:
                methods_count[method] = methods_count.get(method, 0) + 1
        
        self.documentation['summary'] = {
            'total_blueprints': total_blueprints,
            'total_routes': total_routes,
            'methods_distribution': methods_count,
            'blueprints_list': list(self.documentation['blueprints'].keys())
        }
    
    def generate_markdown_report(self) -> str:
        """Generate a markdown documentation report"""
        md = "# Flask Application Documentation\n\n"
        
        # Summary
        summary = self.documentation['summary']
        md += f"## Summary\n"
        md += f"- **Total Blueprints**: {summary['total_blueprints']}\n"
        md += f"- **Total Routes**: {summary['total_routes']}\n"
        md += f"- **HTTP Methods**: {', '.join(f'{k}({v})' for k, v in summary['methods_distribution'].items())}\n\n"
        
        # Blueprint Details
        for bp_name, bp_info in self.documentation['blueprints'].items():
            md += f"## Blueprint: {bp_name}\n"
            md += f"- **URL Prefix**: `{bp_info['url_prefix'] or 'None'}`\n"
            md += f"- **Directory**: `{bp_info['directory']}`\n"
            md += f"- **Routes Count**: {len(bp_info['routes'])}\n"
            
            if bp_info['services_used']:
                md += f"- **Services Used**: {', '.join(bp_info['services_used'])}\n"
            
            md += "\n### Routes\n"
            for route in bp_info['routes']:
                methods_str = ', '.join(route['methods'])
                full_url = f"{bp_info['url_prefix'] or ''}{route['endpoint']}"
                md += f"- **{methods_str}** `{full_url}` -> `{route['function_name']}()`\n"
                
                if route['docstring']:
                    md += f"  - {route['docstring'].split('.')[0]}\n"
                
                if route['parameters']:
                    params = [p for p in route['parameters'] if p not in ['request', 'session']]
                    if params:
                        md += f"  - Parameters: {', '.join(params)}\n"
            
            md += "\n"
        
        return md
    
    def save_documentation(self, output_dir: str = "docs"):
        """Save documentation in multiple formats"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Save JSON
        with open(output_path / "flask_api_documentation.json", 'w') as f:
            json.dump(self.documentation, f, indent=2)
        
        # Save Markdown
        markdown_content = self.generate_markdown_report()
        with open(output_path / "flask_api_documentation.md", 'w', encoding='utf-8') as f:
            f.write(markdown_content)
        
        print(f"Documentation saved to {output_path}/")
        return output_path


def main():
    """Main execution function"""
    # Configure your blueprint folder path here
    BLUEPRINT_FOLDER = "blueprints"  # Adjust this path
    
    if not Path(BLUEPRINT_FOLDER).exists():
        print(f"Blueprint folder '{BLUEPRINT_FOLDER}' not found!")
        return
    
    scanner = FlaskBlueprintScanner(BLUEPRINT_FOLDER)
    documentation = scanner.scan_all_blueprints()
    
    # Print summary
    print("Flask Blueprint Scan Complete!")
    print(f"Found {documentation['summary']['total_blueprints']} blueprints")
    print(f"Found {documentation['summary']['total_routes']} routes")
    
    # Save documentation
    output_path = scanner.save_documentation()
    
    # Print sample of markdown output
    print("\n--- Sample Documentation ---")
    markdown = scanner.generate_markdown_report()
    print(markdown[:500] + "..." if len(markdown) > 500 else markdown)


if __name__ == "__main__":
    main()