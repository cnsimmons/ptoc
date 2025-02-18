'''
import os
import cortex
import sys

def setup_pycortex():
    print(f"Python version: {sys.version}")
    print(f"Pycortex version: {cortex.__version__}")
    print(f"Current working directory: {os.getcwd()}")
    print(f"Pycortex database location: {cortex.database.default_filestore}")
    
    try:
        # Make sure the database directory exists
        db_path = os.path.expanduser('~/.cortex/db')
        os.makedirs(db_path, exist_ok=True)
        print(f"\nCreated/verified database directory at: {db_path}")
        
        print("\nAttempting to download fsaverage template...")
        cortex.utils.download_subject('fsaverage')
        
        # Check both possible locations
        locations = [
            os.path.join(db_path, 'fsaverage'),
            os.path.join(cortex.database.default_filestore, 'fsaverage')
        ]
        
        found = False
        for loc in locations:
            if os.path.exists(loc):
                print(f"\nSuccess! Template found at: {loc}")
                found = True
                break
                
        if not found:
            print("\nWarning: Download completed but template not found in expected locations:")
            for loc in locations:
                print(f"Checked: {loc}")
            
    except Exception as e:
        print(f"\nError during setup: {str(e)}")
        print("\nFull error details:")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    setup_pycortex()
    
'''
import os
import cortex

def explore_database():
    db_path = "/home/csimmon2/anaconda3/envs/fmri/share/pycortex/db"
    print(f"\nExploring pycortex database at: {db_path}")
    
    if not os.path.exists(db_path):
        print("Database directory does not exist!")
        return
        
    # List all contents
    print("\nDatabase contents:")
    for root, dirs, files in os.walk(db_path):
        level = root.replace(db_path, '').count(os.sep)
        indent = ' ' * 4 * level
        print(f"{indent}{os.path.basename(root)}/")
        subindent = ' ' * 4 * (level + 1)
        for f in files:
            print(f"{subindent}{f}")

def main():
    # Print current database info
    print(f"Pycortex database location: {cortex.database.default_filestore}")
    print(f"Does database exist? {os.path.exists(cortex.database.default_filestore)}")
    
    # Explore database contents
    explore_database()

if __name__ == "__main__":
    main()