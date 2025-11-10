import pygame

# === Pygame-Initialisierung ===
pygame.init() # Pygame muss initialisiert sein, um gedrueckte Tasten lesen zu koennen
screen = pygame.display.set_mode((100, 100)) # Kleines, nicht sichtbares Fenster
pygame.display.set_caption("Keyboard reading") # Titel

running = True

while running:
    # --- Pygame-Ereignisverarbeitung ---
    for event in pygame.event.get():
        # Wenn das Fenster geschlossen wird
        if event.type == pygame.QUIT:
            running = False
            print(f"[TEST]: Pygame window closed. Shutting down.")

        # Wenn eine Taste gedrueckt wird
        elif event.type == pygame.KEYDOWN:
            key_name = pygame.key.name(event.key)
            print(f"[TEST]: Key down: {key_name}; {type(key_name)}")

            if event.key == pygame.K_ESCAPE:
                running = False
                print(f"[TEST]: ESC pressed. Shutting down.")

        # Wenn eine Taste losgelassen wird
        elif event.type == pygame.KEYUP:
            key_name = pygame.key.name(event.key)
            print(f"[TEST]: Key up: {key_name}")

    if not running:
        pygame.quit() # Pygame sauber beenden
        break