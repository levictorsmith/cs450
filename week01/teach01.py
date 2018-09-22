import random
import numpy as np

class Movie:
  def __init__(self, title = "", year = "", runtime_min = 0):
    self.title = title
    self.year = year
    self.runtime_min = runtime_min if runtime_min >= 0 else 0
  
  def __repr__(self):
    return self.title + " (" + self.year + ") - " + str(self.runtime_min) + " mins"

  def get_runtime(self):
    hours = self.runtime_min // 60
    minutes = self.runtime_min % 60
    return (hours, minutes)

def create_movie_list():
  return [
    Movie("The Sandlot", "1995", 140),
    Movie("The Shawshank Redemption", "1994", 150),
    Movie("The Godfather", "1972", 160),
    Movie("The Dark Knight", "2008", 170),
    Movie("12 Angry Men", "1957", 180),
  ]

def get_movie_data():
    """
    Generate a numpy array of movie data
    :return:
    """
    num_movies = 10
    array = np.zeros([num_movies, 3], dtype=np.float)

    for i in range(num_movies):
        # There is nothing magic about 100 here, just didn't want ids
        # to match the row numbers
        movie_id = i + 100
        
        # Lets have the views range from 100-10000
        views = random.randint(100, 10000)
        stars = random.uniform(0, 5)

        array[i][0] = movie_id
        array[i][1] = views
        array[i][2] = stars

    return array

def main():
  # Part 1
  first = Movie("The Sandlot", "1995")
  print(first)
  print(first.get_runtime())

  # Part 2
  movies = create_movie_list()
  for movie in movies:
    print(movie)
    pass
  print("")

  # Part 2.3
  long_movies = [movie for movie in movies if movie.runtime_min > 150]
  for movie in long_movies:
    print(movie)
    pass
  print("")

  # Part 2.4
  ratings = {movie.title: random.uniform(0, 5) for movie in movies}
  for key,val in ratings.items():
    print(key, "{:.2f}".format(val))
    pass
  
  # Part 3.1
  movie_data = get_movie_data()
  rows = len(movie_data)
  cols = len(movie_data[0])
  print ("Rows: " + str(rows), "Cols: " + str(cols))

  



if __name__ == '__main__':
  main()