#test maps api
import googlemaps
from datetime import datetime
import os
from pprint import pprint
from datetime import *
from fuzzywuzzy import fuzz
from fuzzywuzzy import process






def dir_deptime(start,dest,dep_time):


    api = os.environ['GOOGLE_API_KEY']
    gmaps = googlemaps.Client(key=api)
    route = gmaps.directions(start,dest,mode="transit",departure_time=dep_time,alternatives=True)

    return route

def dir_arrtime(start,dest,arr_time):


    api = os.environ['GOOGLE_API_KEY']
    gmaps = googlemaps.Client(key=api)
    route = gmaps.directions(start,dest,mode="transit",arrival_time=arr_time,alternatives=True)
    #pprint(route)
    return route



def parse_overral(route):

    distance = route.get("legs")[0].get("distance").get("value")
    duration = timedelta(seconds=route.get("legs")[0].get("duration").get("value"))



    return distance, duration


def parse_steps(route):
    steps = []
    for dict in route.get("legs")[0].get("steps"):
        if(dict.get("travel_mode")=="TRANSIT"):

            if(dict.get("transit_details").get("line").get("vehicle").get("name")!="Bus" and dict.get("transit_details").get("line").get("vehicle").get("name")!="Tram"):
                continue
            dist = dict.get("distance").get("value")
            dur = timedelta(seconds=dict.get("duration").get("value"))
            dep = dict.get("transit_details").get("departure_stop").get("name")
            dep_time = datetime.utcfromtimestamp(dict.get("transit_details").get("departure_time").get("value"))
            arr = dict.get("transit_details").get("arrival_stop").get("name")
            arr_time = datetime.utcfromtimestamp(dict.get("transit_details").get("arrival_time").get("value"))
            towards = dict.get("transit_details").get("headsign")
            line = dict.get("transit_details").get("line").get("short_name")
            stops = dict.get("transit_details").get("num_stops")
            type = dict.get("travel_mode")
            steps.append({"type":type,"dist":dist,"dur":dur,"dep":dep,"dep_time":dep_time,
            "arr":arr,"arr_time":arr_time,"line":line,"towards":towards,"stops":stops})

        else:
            dist = dict.get("distance").get("value")
            dur = timedelta(seconds=dict.get("duration").get("value"))
            type = dict.get("travel_mode")
            instruction = dict.get("html_instructions")
            steps.append({"type":type,"instruction":instruction,"dist":dist,"dur":dur})

    return steps





halteList = [line.rstrip('\n') for line in open("../data/vbz_fahrgastzahlen/stationen.txt")]
#print(halteList)

hour =11
minute = 15
dt = datetime.now().replace(hour=hour,minute=minute,day=6) #monday
start = "Zehntenhausplatz,Zürich"
destination = "ETH Zürich"
now = datetime.now()

routes = []






for i in range (0,120,30):


    route= dir_arrtime(start,destination,dt-timedelta(minutes=i))
    for j in range(0,len(route)):
        r=[parse_overral(route[j]),parse_steps(route[j]),0]

        if r not in routes:
            routes.append(r)

for r in range(0,len(routes)):
    for j in range(len(routes[r][1])):
        if(routes[r][1][j].get("type")=="TRANSIT"):


            dep = process.extractOne(routes[r][1][j].get("dep"),halteList)[0]
            
            print(routes[r][1][j].get("arr"))
            arr = process.extractOne(routes[r][1][j].get("arr"),halteList)[0]
            print(arr)


    print()




#pprint(routes)
