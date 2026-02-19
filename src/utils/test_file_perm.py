import win32security
import ntsecuritycon as con
def secure_file_acl(FILE_PATH: str, TARGET_USER: str):
    # Lookup SIDs
    user_sid, _, _ = win32security.LookupAccountName(None, TARGET_USER)
    admin_sid, _, _ = win32security.LookupAccountName(None, "Administrator")

    # Build security descriptor
    sd = win32security.SECURITY_DESCRIPTOR()
    dacl = win32security.ACL()

    # Owner = Administrator
    sd.SetSecurityDescriptorOwner(admin_sid, False)

    # Target user: Read + Write
    dacl.AddAccessAllowedAce(
        win32security.ACL_REVISION,
        con.FILE_GENERIC_READ | con.FILE_GENERIC_WRITE,
        user_sid
    )

    # Apply DACL (disable inheritance)
    sd.SetSecurityDescriptorDacl(True, dacl, False)

    win32security.SetFileSecurity(
        FILE_PATH,
        win32security.DACL_SECURITY_INFORMATION,
        sd
    )

    print("Owner=Administrator, only TARGET_USER has RW")

