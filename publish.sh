#!/bin/bash
# Скрипт для публикации PlanetPattern на GitHub

echo "🚀 Публикация PlanetPattern на GitHub"
echo ""

# Проверка git
if ! git rev-parse --git-dir > /dev/null 2>&1; then
    echo "❌ Git не инициализирован"
    exit 1
fi

echo "✅ Git инициализирован"

# Проверка remote
if git remote | grep -q origin; then
    echo "✅ Remote уже настроен:"
    git remote -v
    echo ""
    read -p "Отправить на GitHub? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        git push -u origin main
        echo ""
        echo "✅ Отправлено!"
    fi
else
    echo "⚠️ Remote не настроен"
    echo ""
    echo "1. Сначала создай репозиторий на GitHub:"
    echo "   https://github.com/new"
    echo "   Имя: PlanetPattern"
    echo ""
    echo "2. Затем выполни:"
    echo "   git remote add origin https://github.com/USERNAME/PlanetPattern.git"
    echo "   git push -u origin main"
fi

