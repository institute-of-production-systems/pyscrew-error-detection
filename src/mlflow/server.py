import atexit
import os
import subprocess
import sys
import time

from src.utils import get_logger

# Store the mlflow process globally so we can terminate it properly
_mlflow_process = None


def launch_server(port=5000, wait=True):
    """
    Launch the MLflow server programmatically with consistent logging.

    Args:
        port (int): Port to run the server on
        wait (bool): If True, wait for server to start before returning

    Returns:
        subprocess.Popen: The process object for the server
    """
    global _mlflow_process

    # Set up logger with the same format as your application
    logger = get_logger(__name__)

    # If server is already running, return the process
    if _mlflow_process is not None and _mlflow_process.poll() is None:
        logger.info(f"MLflow server already running at http://127.0.0.1:{port}")
        return _mlflow_process

    # Get the directory where this module is located
    module_dir = os.path.dirname(os.path.abspath(__file__))

    # Determine which script to run based on the OS
    if sys.platform.startswith("win"):
        script_path = os.path.join(module_dir, "launch_server.bat")
        # On Windows, we need shell=True to execute batch files
        logger.info(f"Starting MLflow server on port {port}...")
        _mlflow_process = subprocess.Popen(
            f"{script_path} {port}",
            shell=True,
            cwd=module_dir,
            stdout=subprocess.PIPE,  # Capture output so we can log it consistently
            stderr=subprocess.STDOUT,
            text=True,
        )
    else:
        script_path = os.path.join(module_dir, "launch_server.sh")
        # Make sure the script is executable
        os.chmod(script_path, 0o755)  # rwxr-xr-x
        logger.info(f"Starting MLflow server on port {port}...")
        _mlflow_process = subprocess.Popen(
            [script_path, str(port)],
            cwd=module_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )

    # Register function to terminate server on exit
    atexit.register(stop_server)

    if wait:
        # Wait a moment for the server to start
        # But also grab any initial output and log it in our format
        startup_timeout = 5  # seconds
        start_time = time.time()

        def read_process_output(timeout):
            """Read process output with timeout"""
            output_lines = []
            while time.time() - start_time < timeout:
                if _mlflow_process.poll() is not None:
                    # Process ended
                    break

                line = _mlflow_process.stdout.readline().strip()
                if line:
                    # Add to output lines and log using our logger
                    output_lines.append(line)
                    # Skip lines that might already have timestamps
                    if not line.startswith("20"):  # Doesn't start with year
                        logger.info(line)

                if "Serving on http://127.0.0.1:" in line:
                    # Server is up
                    break

                time.sleep(0.1)

            return output_lines

        # Read initial output during startup
        startup_output = read_process_output(startup_timeout)

        # Check if server actually started
        if any("Serving on http://127.0.0.1:" in line for line in startup_output):
            logger.info(f"MLflow UI available at http://127.0.0.1:{port}")
        else:
            # Server might be slow to start, so we'll just wait
            time.sleep(3)
            logger.info(f"MLflow UI available at http://127.0.0.1:{port}")

        # Start a background thread to continue capturing and logging output
        import threading

        def log_process_output():
            """Continuously log process output until it ends"""
            while _mlflow_process.poll() is None:
                line = _mlflow_process.stdout.readline().strip()
                if line:
                    # Skip lines that might already have timestamps
                    if not line.startswith("20"):  # Doesn't start with year
                        logger.info(line)
                else:
                    # No more output but process still running
                    time.sleep(0.1)

        output_thread = threading.Thread(target=log_process_output)
        output_thread.daemon = True  # Thread will die when main program exits
        output_thread.start()

    return _mlflow_process


def stop_server():
    """Stop the MLflow server if it's running."""
    global _mlflow_process

    # Set up logger
    logger = logger = get_logger(__name__)

    if _mlflow_process is not None:
        if _mlflow_process.poll() is None:  # If process is still running
            logger.info("Stopping MLflow server...")
            _mlflow_process.terminate()
            _mlflow_process.wait(timeout=5)
            logger.info("MLflow server stopped")
        _mlflow_process = None
