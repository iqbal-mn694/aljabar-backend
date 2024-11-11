from django.http import HttpResponse
from tubesAljabar.services.recommender.main import main

def recommendSong(request):
  return  HttpResponse(main(request), content_type="application/json")
                                           
