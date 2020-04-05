
import googlemaps
from datetime import datetime
import os
from pprint import pprint
from datetime import *
from fuzzywuzzy import fuzz
from fuzzywuzzy import process
import csv
import numpy as np


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

            if(dict.get("transit_details").get("line").get("vehicle").get("name")!="Tram" and dict.get("transit_details").get("line").get("vehicle").get("name")!="Bus"):

                return[]
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


halteList = [line.rstrip('\n') for line in open("../data/vbz_fahrgastzahlen/stationen.txt")] #data bus and tram stations
#print(halteList)
capacities = {32:{"seats":60,"stands":95,"overall":155},
61:{"seats":43,"stands":54,"overall":97},
62:{"seats":43,"stands":54,"overall":97},
10:{"seats":90,"stands":130,"overall":220},
6:{"seats":90,"stands":130,"overall":220},
15:{"seats":48,"stands":72,"overall":120},
11:{"seats":90,"stands":130,"overall":220}}

with open('../data/vbz_fahrgastzahlen/Haltestellen_Richtungen.csv', newline='') as f:
    reader = csv.reader(f)
    endstations = list(reader)
endstations=endstations[1:][:]

dirs={}                                          #directions mapped to endstations
for i in range(len(endstations)):
    dirs[endstations[i][4]]=endstations[i][1]







start = input("From? ")
destination = input("To? ")
h = int(input("Hour? "))
m = int(input("Minute "))
timebefore = int(input("Time flexibility in mins? "))
"""
hour =11
minute = 15
timebefore = 120 #time free before arrival time (in mins)
start = "Zehntenhausplatz,Zürich"
destination = "ETH Zürich"
"""
dt = datetime.now().replace(hour=h+2,minute=m,day=6) #monday

now = datetime.now()

routes = []

for i in range (0,timebefore,30):


    route= dir_arrtime(start,destination,dt-timedelta(minutes=i))   #dir_arrtime for arrivaltime; dir_deptime for deptime
    for j in range(0,len(route)):
        r=[parse_overral(route[j]),parse_steps(route[j]),0] #[overall route infos,steps infos,rating]
        #pprint(r)
        if r not in routes:
            routes.append(r)
#pprint(routes)
print()
print("Calculating ",len(routes), "possible routes...")
print()
for r in range(0,len(routes)):
    ratio=0
    count=0
    for j in range(len(routes[r][1])):

        if(routes[r][1][j].get("type")=="TRANSIT"):
            count+=1
            dep = process.extractOne(routes[r][1][j].get("dep"),halteList)[0]
            dep_time=routes[r][1][j].get("dep_time")
            towards = routes[r][1][j].get("towards")
            line = routes[r][1][j].get("line")

            dir = dirs.get(process.extractOne(towards,halteList)[0])
            print(dep,dep_time,"Line: ",line, "towards: ",dir)
            #pred_dep =predict_besetzung(dep_time, line, dep, dir)
            ratio_dep =1#capacities.get(32).get("overall")
            #print(ratio_dep)

            arr = process.extractOne(routes[r][1][j].get("arr"),halteList)[0]
            arr_time=routes[r][1][j].get("arr_time")
            print(arr,arr_time)

            #pred_arr = predict_besetzung(arr_time, line, arr, dir)
            ratio_arr = 1# pred_dep[0]/capacities.get(32).get("overall")
            #print(ratio_arr)
            ratio+=(ratio_dep+ratio_arr)

    if(count!=0):
        ratio/=count*2
        routes[r][2]=ratio
        print(ratio)
    print()
