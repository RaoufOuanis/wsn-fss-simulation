import os
import subprocess
 
def main():
    # simple helper: `python scripts/run_app.py`
    this_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(this_dir, ".."))
    app_path = os.path.join(project_root, "app", "app_streamlit.py")

    env = os.environ.copy()
    python_path = env.get("PYTHONPATH")
    # make sure the project package is importable for the Streamlit subprocess
    env["PYTHONPATH"] = os.pathsep.join(
        [p for p in [project_root, python_path] if p]
    )

    subprocess.run(["streamlit", "run", app_path], cwd=project_root, env=env)


if __name__ == "__main__":
    main()
