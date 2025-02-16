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