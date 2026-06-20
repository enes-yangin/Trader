import warnings
import os
import sys

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

def load_env():
    paths = [".env"]
    if getattr(sys, "frozen", False):
        exe_dir = os.path.dirname(sys.executable)
        paths.append(os.path.join(exe_dir, ".env"))
    
    for path in paths:
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line or line.startswith("#"):
                            continue
                        if "=" in line:
                            k, v = line.split("=", 1)
                            k = k.strip()
                            v = v.strip().strip('"').strip("'")
                            os.environ[k] = v
                break
            except Exception:
                pass

load_env()

if __name__ == "__main__":
    try:
        from ui.app import App
        app = App()
        app.mainloop()
    except Exception as e:
        import traceback
        try:
            import tkinter as tk
            from tkinter import messagebox
            root = tk.Tk()
            root.withdraw()
            messagebox.showerror(
                "TraderAI Startup Error",
                f"Application failed to initialize.\n\nError: {e}\n\nTraceback:\n{traceback.format_exc()}"
            )
        except Exception:
            print("TraderAI Startup Error:", e)
            traceback.print_exc()
