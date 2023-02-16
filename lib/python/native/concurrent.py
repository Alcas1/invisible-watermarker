import concurrent.futures


class Concurrent:

  @staticmethod
  def execute_concurrently(function, kwargs_list):
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
      futures = [executor.submit(function, kwargs) for kwargs in kwargs_list]
    for future in concurrent.futures.as_completed(futures):
      results.append(future.result())
    return results
