# 🚀 Настройка GitHub-репозитория

## Шаг 1: Создать репозиторий на GitHub

1. Зайди на https://github.com/new
2. Имя репозитория: `PlanetPattern`
3. Описание: "Experiments in rhythmic, self-adapting AI — Physics of Living Systems"
4. Публичный, с README (можно оставить пустым)

## Шаг 2: Локальная инициализация

```bash
cd /Users/sanecek/tema/fractal-ai/planet_pattern

# Инициализация git
git init
git add .
git commit -m "Initial commit — Planet Pattern v2: rhythmic AI prototype"

# Подключение к GitHub (замени USERNAME на свой)
git remote add origin https://github.com/USERNAME/PlanetPattern.git
git branch -M main
git push -u origin main
```

## Шаг 3: Проверка

После push проверь:
- ✅ README.md отображается
- ✅ LICENSE видна
- ✅ Структура папок правильная
- ✅ requirements.txt есть

## Шаг 4: Добавить теги и release

```bash
git tag -a v2.0 -m "Planet Pattern v2 — Rhythmic AI prototype"
git push origin v2.0
```

## Шаг 5: Добавить topics (на GitHub UI)

- `artificial-intelligence`
- `rhythmic-learning`
- `wavelet-memory`
- `physics-of-living-systems`
- `research-prototype`
- `python`

## Структура файлов для GitHub

```
PlanetPattern/
├── dashboard/
│   └── planet_pattern_app.py     ✅
├── docs/
│   └── technologies_of_new_civilization.md  ✅
├── examples/
│   └── test_run.py               ✅
├── README.md                      ✅
├── requirements.txt               ✅
├── LICENSE                        ✅
└── .gitignore                     ✅
```

---

**Готово! Репозиторий готов к публикации.** 🎉

