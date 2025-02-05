import subprocess
import os
from setup_test_env import setup_test_environment

def run_tests():
    # Setup test environment
    setup_test_environment()
    
    # Run behave with HTML formatter
    cmd = [
        'behave',
        '--format=behave_html_formatter:HTMLFormatter',
        '--outfile=test-reports/behave-report.html',
        '--format=pretty',  # Also show console output
        'features/'
    ]
    
    try:
        subprocess.run(cmd, check=True)
        print("\nTest execution completed. HTML report generated at test-reports/behave-report.html")
        
        # Make the report more readable by adding CSS
        report_file = 'test-reports/behave-report.html'
        if os.path.exists(report_file):
            with open(report_file, 'r') as f:
                content = f.read()
            
            # Add CSS for better styling
            css = '''
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                .feature { margin-bottom: 30px; }
                .scenario { margin: 20px 0; padding: 10px; background: #f5f5f5; }
                .passed { color: green; }
                .failed { color: red; }
                .skipped { color: orange; }
                .step { margin: 5px 0; }
                .description { font-style: italic; color: #666; }
            </style>
            '''
            
            content = content.replace('</head>', f'{css}</head>')
            
            with open(report_file, 'w') as f:
                f.write(content)
    except subprocess.CalledProcessError as e:
        print(f"\nTest execution failed with error code {e.returncode}")
        raise

if __name__ == '__main__':
    run_tests() 