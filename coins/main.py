import pygame
import random

# Инициализация
pygame.init()

# Константы
WIDTH, HEIGHT = 800, 600
PLAYER_WIDTH, PLAYER_HEIGHT = 80, 20
ITEM_SIZE = 30
PLAYER_SPEED = 7
ITEM_SPEED = 5

# Цвета
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)

# Создание окна
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Catch the Falling Objects")
clock = pygame.time.Clock()
font = pygame.font.Font(None, 36)

# Игрок
player = pygame.Rect(WIDTH // 2 - PLAYER_WIDTH // 2, HEIGHT - 30,
                     PLAYER_WIDTH, PLAYER_HEIGHT
                     )

# Предметы
items = []
score = 0
running = True

# Основной цикл
while running:
    # Обработка событий
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Управление
    keys = pygame.key.get_pressed()
    if keys[pygame.K_LEFT] and player.left > 0:
        player.x -= PLAYER_SPEED
    if keys[pygame.K_RIGHT] and player.right < WIDTH:
        player.x += PLAYER_SPEED

    # Создание предметов
    if random.randint(1, 30) == 1:
        items.append(pygame.Rect(random.randint(0, WIDTH - ITEM_SIZE),
                                 0, ITEM_SIZE, ITEM_SIZE))

    # Движение предметов
    for item in items[:]:
        item.y += ITEM_SPEED

        # Проверка столкновения
        if player.colliderect(item):
            items.remove(item)
            score += 1

        # Удаление за пределами экрана
        elif item.y > HEIGHT:
            items.remove(item)

    # Отрисовка
    screen.fill(BLACK)
    pygame.draw.rect(screen, BLUE, player)

    for item in items:
        pygame.draw.rect(screen, RED, item)

    # Отображение счета
    score_text = font.render(f"Score: {score}", True, WHITE)
    screen.blit(score_text, (10, 10))

    pygame.display.flip()
    clock.tick(60)

pygame.quit()