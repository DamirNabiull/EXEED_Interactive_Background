По установке:

Нулевое - установить vlc плеер - https://www.videolan.org/vlc/index.ru.html

Первое - скачать и установить CUDA - это один из самых долгих процессов
https://developer.nvidia.com/cuda-downloads

Второе - скачать и установить python с оф сайта и поставить галочку в строке с PATH на первом же экране.
https://www.python.org/downloads/

Третье - в папке Installation скрипт python.cmd - его запустить, тоже долго ставиться будет

design.cmd - НЕ ЗАПУСКАТЬ (Лучше вообще сразу удалить) - там пути к папкам немного собьются

Четвертое - в папке Application/image удалить картинку background.jpg.

Пятое - в папке Application запустить start_service.cmd

Шестое - в папке Application запустить start_gui.cmd, билд должен запуститься и закрыться секунд через 20.

Седьмое - повторить шестой шаг - все должно заработать