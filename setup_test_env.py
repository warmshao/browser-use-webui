import os

def setup_test_environment():
    # Create test reports directory if it doesn't exist
    reports_dir = 'test-reports'
    if not os.path.exists(reports_dir):
        os.makedirs(reports_dir)

if __name__ == '__main__':
    setup_test_environment() 