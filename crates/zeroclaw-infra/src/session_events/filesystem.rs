//! Plaintext session content uses private local filesystem objects. Do not
//! follow a symlink for event files; successful commits still require fsync.
use super::*;
use std::fs::{File, OpenOptions};
use std::io::Read;

/// Sync control-file publication using the same platform contract as config
/// publication. Windows cannot flush directory handles (ERROR_ACCESS_DENIED);
/// files must be synced before rename. JSONL commits use their stricter path.
pub(super) fn sync_directory(path: &Path) -> Result<(), AppendError> {
    #[cfg(unix)]
    {
        File::open(path)
            .and_then(|directory| directory.sync_all())
            .map_err(jsonl::storage)
    }
    #[cfg(windows)]
    {
        use std::os::windows::fs::OpenOptionsExt;
        const FILE_FLAG_BACKUP_SEMANTICS: u32 = 0x0200_0000;
        let directory = OpenOptions::new()
            .read(true)
            .custom_flags(FILE_FLAG_BACKUP_SEMANTICS)
            .open(path)
            .map_err(jsonl::storage)?;
        match directory.sync_all() {
            Ok(()) => Ok(()),
            Err(error) if error.raw_os_error() == Some(5) => Ok(()),
            Err(error) => Err(jsonl::storage(error)),
        }
    }
    #[cfg(not(any(unix, windows)))]
    {
        let _ = path;
        Err(AppendError::StorageUnavailable(
            "session publication unsupported on this platform".into(),
        ))
    }
}

/// Read access never creates a missing object or follows a substituted link.
/// Validate the opened descriptor, so a FIFO cannot become a blocking reader.
pub(super) fn reader(path: &Path) -> std::io::Result<File> {
    let mut options = OpenOptions::new();
    options.read(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK);
    }
    let file = options.open(path)?;
    if !file.metadata()?.is_file() {
        return Err(std::io::Error::other("session file is not a regular file"));
    }
    Ok(file)
}

pub(super) fn read(path: &Path) -> std::io::Result<Vec<u8>> {
    let mut bytes = Vec::new();
    reader(path)?.read_to_end(&mut bytes)?;
    Ok(bytes)
}

/// SQLite owns ordinary handles to its database and sidecars. Closing another
/// ordinary handle can release all of SQLite's process-wide POSIX locks.
/// Permission repair must therefore avoid opening those files for I/O.
pub(super) fn private_sqlite_file(path: &Path) -> std::io::Result<()> {
    #[cfg(target_os = "linux")]
    {
        use std::os::{
            fd::AsRawFd,
            unix::fs::{OpenOptionsExt, PermissionsExt},
        };
        // O_PATH pins the checked inode without an I/O handle. Linux does not
        // remove POSIX record locks when an O_PATH descriptor is closed.
        let file = OpenOptions::new()
            .read(true)
            .custom_flags(libc::O_PATH | libc::O_NOFOLLOW)
            .open(path)?;
        let metadata = file.metadata()?;
        if !metadata.is_file() {
            return Err(std::io::Error::other("session file is not a regular file"));
        }
        if metadata.permissions().mode() & 0o7777 != 0o600 {
            // This is the kernel-owned link for the live pinned descriptor,
            // not a lookup through a replaceable user-supplied symlink.
            fs::set_permissions(
                format!("/proc/self/fd/{}", file.as_raw_fd()),
                fs::Permissions::from_mode(0o600),
            )?;
        }
        Ok(())
    }
    #[cfg(not(target_os = "linux"))]
    {
        let metadata = fs::symlink_metadata(path)?;
        if !metadata.is_file() || metadata.file_type().is_symlink() {
            return Err(std::io::Error::other("session file is not a regular file"));
        }
        #[cfg(unix)]
        {
            use std::os::unix::{ffi::OsStrExt, fs::PermissionsExt};
            if metadata.permissions().mode() & 0o7777 != 0o600 {
                let path =
                    std::ffi::CString::new(path.as_os_str().as_bytes()).map_err(|error| {
                        std::io::Error::new(std::io::ErrorKind::InvalidInput, error)
                    })?;
                // SAFETY: path is NUL-terminated and valid for the duration
                // of this call. Never follow a substituted final symlink.
                if unsafe {
                    libc::fchmodat(
                        libc::AT_FDCWD,
                        path.as_ptr(),
                        0o600,
                        libc::AT_SYMLINK_NOFOLLOW,
                    )
                } != 0
                {
                    return Err(std::io::Error::last_os_error());
                }
            }
        }
        Ok(())
    }
}

pub(super) fn directory(path: &Path) -> Result<(), AppendError> {
    fs::create_dir_all(path).map_err(jsonl::storage)?;
    let metadata = fs::symlink_metadata(path).map_err(jsonl::storage)?;
    if !metadata.is_dir() || metadata.file_type().is_symlink() {
        return Err(AppendError::StorageUnavailable(
            "session directory is not a real directory".into(),
        ));
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        fs::set_permissions(path, fs::Permissions::from_mode(0o700)).map_err(jsonl::storage)?;
    }
    Ok(())
}
pub(super) fn file(path: &Path, exclusive: bool) -> std::io::Result<File> {
    let mut options = OpenOptions::new();
    options.read(true).write(true);
    if exclusive {
        options.create_new(true);
    } else {
        options.create(true).truncate(false);
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600).custom_flags(libc::O_NOFOLLOW);
    }
    let file = options.open(path)?;
    if !file.metadata()?.is_file() {
        return Err(std::io::Error::other("session file is not a regular file"));
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        file.set_permissions(fs::Permissions::from_mode(0o600))?;
    }
    Ok(file)
}
