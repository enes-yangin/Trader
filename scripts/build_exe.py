import os
import sys
import shutil

def main():
    print("=== TraderAI Standalone Executable Builder ===")
    
    # Force close any running instances of the app to release file locks
    try:
        if os.name == "nt":
            os.system("taskkill /f /im TraderAI.exe >nul 2>nul")
    except Exception:
        pass
        
    # 1. Clean previous builds
    for path in ["build", "dist"]:
        if os.path.exists(path):
            print(f"Cleaning existing {path} folder...")
            try:
                shutil.rmtree(path)
            except Exception as e:
                try:
                    import random
                    temp_path = f"{path}_old_{random.randint(1000, 9999)}"
                    os.rename(path, temp_path)
                    shutil.rmtree(temp_path, ignore_errors=True)
                except Exception:
                    print(f"\nERROR: Could not clean directory '{path}'. Is TraderAI.exe still running?")
                    print("Please close any running instances of TraderAI and try again.")
                    print(f"Details: {e}")
                    sys.exit(1)
            
    # 2. PyInstaller command arguments
    pyinstaller_args = [
        "main.py",
        "--name=TraderAI",
        "--onedir",
        "--clean",
        "--noconfirm",
        "--noconsole",
        "--collect-data=xgboost",
        "--collect-binaries=xgboost",
        "--collect-data=vaderSentiment",
        "--collect-data=transformers",
        "--collect-data=optuna",
        "--collect-data=ta",
        "--collect-data=pyarrow",
        "--hidden-import=sklearn.utils._typedefs",
        "--hidden-import=sklearn.neighbors._typedefs",
        "--hidden-import=sklearn.neighbors._quad_tree",
        "--hidden-import=sklearn.tree._utils",
        "--hidden-import=scipy.special.cython_special",
        "--hidden-import=scipy.integrate",
    ]
    
    # Run PyInstaller
    import PyInstaller.__main__
    print("Starting PyInstaller compilation (this may take several minutes)...")
    try:
        PyInstaller.__main__.run(pyinstaller_args)
        print("\n=== BUILD SUCCESSFUL ===")
        print("The standalone package is generated at: dist/TraderAI")
        print("Double-click dist/TraderAI/TraderAI.exe to run the application.")
    except Exception as e:
        print(f"\n=== BUILD FAILED ===\nError: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
