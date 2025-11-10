import pygame

# === Pygame initialisation ===
pygame.init() # Pygame must be initialized to read pressed keys
screen = pygame.display.set_mode((100, 100)) # Small, not visible window
pygame.display.set_caption("Keyboard reading") # Title

running = True

while running:
    # --- Pygame event processing ---
    for event in pygame.event.get():
        # If the window is closed
        if event.type == pygame.QUIT:
            running = False
            print(f"[TEST]: Pygame window closed. Shutting down.")

        # When a key is pressed
        elif event.type == pygame.KEYDOWN:
            key_name = pygame.key.name(event.key)
            print(f"[TEST]: Key down: {key_name}; {type(key_name)}")

            if event.key == pygame.K_ESCAPE:
                running = False
                print(f"[TEST]: ESC pressed. Shutting down.")

        # When a key is released
        elif event.type == pygame.KEYUP:
            key_name = pygame.key.name(event.key)
            print(f"[TEST]: Key up: {key_name}")

    if not running:
        pygame.quit() # Pygame sauber beenden
        break