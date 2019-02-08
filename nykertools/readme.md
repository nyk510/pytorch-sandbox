# Nyker Tools

汎用的に使えそうなコードをまとめるモジュール

docker-compose で起動していればこのモジュールには python path が通っているためどこからでも import 出来る

```python
from nykertools import utils

@utils.stopwatch
def my_function():
  x = 0
  for i in range(1000):
    x += i
  return x

my_function()
```