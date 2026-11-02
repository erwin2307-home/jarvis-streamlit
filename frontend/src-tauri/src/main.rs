#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use tauri::{Emitter, Manager};
use tauri_plugin_global_shortcut::{Code, GlobalShortcutExt, Modifiers, Shortcut, ShortcutState};

fn main() {
    let shortcut = Shortcut::new(Some(Modifiers::SUPER), Code::KeyJ);

    tauri::Builder::default()
        .plugin(
            tauri_plugin_global_shortcut::Builder::new()
                .with_handler(move |app, sc, event| {
                    if sc == &shortcut && event.state() == ShortcutState::Pressed {
                        if let Some(win) = app.get_webview_window("main") {
                            let _ = win.emit("hotkey", "manual_start");
                        }
                    }
                })
                .build(),
        )
        .setup(move |app| {
            // Tauri 2 dev-mode hot reloads (or any prior crashed instance) can leave
            // Win+J registered with the OS. Unregister defensively before register,
            // and tolerate the "not registered" error from a fresh boot.
            let gs = app.global_shortcut();
            let _ = gs.unregister(shortcut);
            if let Err(e) = gs.register(shortcut) {
                eprintln!("warning: failed to register Win+J global shortcut: {e}");
                eprintln!("  hotkey will not work; HUD click + voice still operational");
            }
            Ok(())
        })
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
