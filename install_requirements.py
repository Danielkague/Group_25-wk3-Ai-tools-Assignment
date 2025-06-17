import subprocess
import sys
import pkg_resources

def install_requirements():
    print("Installing required packages...")
    
    # Read requirements from requirements.txt
    with open('requirements.txt', 'r') as f:
        requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]
    
    # Install each requirement
    for requirement in requirements:
        try:
            # Check if package is already installed
            pkg_name = requirement.split('>=')[0].strip()
            try:
                pkg_resources.require(pkg_name)
                print(f"✓ {pkg_name} is already installed")
                continue
            except (pkg_resources.DistributionNotFound, pkg_resources.VersionConflict):
                pass
            
            print(f"Installing {requirement}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", requirement])
            print(f"✓ Successfully installed {requirement}")
            
        except subprocess.CalledProcessError as e:
            print(f"✗ Failed to install {requirement}")
            print(f"Error: {str(e)}")
            return False
    
    print("\nAll requirements have been installed successfully!")
    return True

if __name__ == "__main__":
    install_requirements() 