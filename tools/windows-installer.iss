; The Windows installer, filled in by tools/package-studio.py
; (windows-installer) and compiled by Inno Setup's ISCC. The words with
; a dollar sign are the tool's; {braces} and {{ are Inno Setup's own.
[Setup]
AppId={{$app_id}
AppName=$name
AppVersion=$version
AppVerName=$name $version
VersionInfoVersion=$numeric
AppPublisher=$publisher
AppPublisherURL=$homepage
AppSupportURL=$homepage/issues
DefaultDirName={localappdata}\Programs\$name
DisableProgramGroupPage=yes
DisableDirPage=yes
PrivilegesRequired=lowest
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
MinVersion=10.0
OutputBaseFilename=$name-setup
Compression=lzma2/ultra64
SolidCompression=yes
WizardStyle=modern
SetupIconFile=$name.ico
UninstallDisplayIcon={app}\$name.ico
UninstallDisplayName=$name
CloseApplications=yes

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked

[Files]
Source: "$exe"; DestDir: "{app}"; Flags: ignoreversion
Source: "$name.ico"; DestDir: "{app}"; Flags: ignoreversion
Source: "LICENSE"; DestDir: "{app}"; Flags: ignoreversion
Source: "NOTICE"; DestDir: "{app}"; Flags: ignoreversion
Source: "LICENSES\*"; DestDir: "{app}\LICENSES"; Flags: ignoreversion recursesubdirs
$third_party
[Icons]
Name: "{autoprograms}\$name"; Filename: "{app}\$exe"; IconFilename: "{app}\$name.ico"
Name: "{autodesktop}\$name"; Filename: "{app}\$exe"; IconFilename: "{app}\$name.ico"; Tasks: desktopicon

[Run]
Filename: "{app}\$exe"; Description: "{cm:LaunchProgram,$name}"; Flags: nowait postinstall skipifsilent
