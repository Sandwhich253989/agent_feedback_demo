import subprocess
from pathlib import Path
# from set_logging import logger

def secure_file_acl(filepath: Path,TARGET_USER: str):
    try:
        subprocess.run(
            ["icacls", str(filepath), "/inheritance:r"],
            check=True
        )

        # Grant full control to target user
        subprocess.run(
            ["icacls", str(filepath), f"/grant:r", f"{TARGET_USER}:(F)"],
            check=True
        )
        #
        # # Grant full control to Administrators
        # subprocess.run(
        #     ["icacls", str(filepath), f"/grant", f"Administrators:(F)"],
        #     check=True
        # )

        # Explicitly remove Guests / Everyone
        subprocess.run(
            ["icacls", str(filepath), "/remove", "Guests", "Everyone"],
            check=False
        )

        print(f"[INFO]  FILE PERMISSIONS SET for {TARGET_USER} ")
    except Exception as e:
        # logger.error(f"[ERROR] {e}")
        print(f"[ERROR] UNABLE TO SET FILE PERMISSIONS for {TARGET_USER} : {e}")

    return

# secure_file_acl('../outputs/output_test.docx',"Administrators")