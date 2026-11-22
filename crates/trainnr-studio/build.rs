//! Windows: the exe's version resource, which Properties, Details, Task
//! Manager and the firewall prompt read. Without one, v0.2.0's
//! `trainnr-studio.exe` showed no version and no name of its own there
//! (Windows 11, 2026-11-18). The versions are the package's; the release
//! checks the built exe carries them (`tools/package-studio.py
//! windows-version-check`). Other targets: nothing.

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() != Ok("windows") {
        return;
    }
    let mut res = winresource::WindowsResource::new();
    res.set("ProductName", "trainnr")
        .set("FileDescription", "trainnr Studio")
        .set("CompanyName", "Prakhar Aggarwal")
        .set(
            "LegalCopyright",
            "Copyright 2026 Prakhar Aggarwal. Apache-2.0.",
        );
    if let Err(e) = res.compile() {
        // A build without the resource compiler (the Windows SDK's rc.exe)
        // stops here rather than ship an exe that names no version.
        panic!("the Windows version resource: {e}");
    }
}
