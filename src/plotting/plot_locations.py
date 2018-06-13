# -*- coding: utf-8 -*-
"""
Created on Thu Jun  7 11:33:29 2018

@author: Joris
"""
import xml.etree.ElementTree as ET
from pyproj import Proj
import pyproj
import os
import datetime
from operator import itemgetter
import cv2

class PlotEPSGcoordinates(object):
    
    def __init__(self,PATH_TO_VIDEO, epsg = 28992):
        
        self.PATH_TO_VIDEO = PATH_TO_VIDEO
        self.p1 = Proj(init = 'epsg:'+str(epsg))
        self.p2 = Proj(init='epsg:4326')

        # Retrieve all the XML files
        self.files = [file_name for file_name in os.listdir(self.PATH_TO_VIDEO) 
                if file_name.endswith('.xml')]
        
        self.sorted_files = self.sort_files()
        
    def sort_files(self):
        """ Function that returns a list of XML filenames, sorted by date
        """
        
        sorted_files = []
        unsorted_dates = []
        
        for file in self.files:
            root = ET.parse(os.path.join(self.PATH_TO_VIDEO,file)).getroot()  
            
            # Retrieve the date objects
            date = root.attrib['recording-date']
            unsorted_dates.append(date)
            
            date,time = date.split('T')[0], date.split('T')[1]
            
            date = date.split('-')
            year = int(date[0])
            month = int(date[1])
            day = int(date[2])
        
            time = time.split(':')
            hour = int(time[0])
            minute = int(time[1])
            second = int(time[2][0:2])
            
            msecond = time[2].replace(str(second)+'.',"")
            msecond = msecond.split('+')[0]
            
            if len (msecond) == 1:
                msecond = int(msecond) * 10
            else:
                msecond = int(msecond)
                
            msecond *= 10000
            
            time_obj = datetime.datetime(year,month,day,hour,minute,second,msecond)
            
            x1 = root.attrib['recording-location-x']
            y1 = root.attrib['recording-location-y']
            z1 = root.attrib['recording-location-z']
            
            lat,long = self.convert_coordinates(x1,y1,z1)
            
            file_name = file.replace('.xml','.jpg')
            
            sorted_files.append((time_obj,file_name,lat,long))
        
        # Sort by the time object
        sorted_files.sort(key=itemgetter(0))
                
        return sorted_files
    
    def convert_coordinates(self,x,y,z):
        """ Function that converts the EPSG:28992 coordinates to EPSG:4236
        """
        
        long, lat, _ = pyproj.transform(self.p1, self.p2, x, y, z, radians=False)
        return lat, long
    
    def make_video(self, video_name = 'video.avi'):
        """ Given the sorted_files list, convert it into a video
        """
        
        # List of '.jpg' files
        images = [data_point[1] for data_point in self.sorted_files]
        
        # Define the first frame
        frame = cv2.imread(os.path.join(self.PATH_TO_VIDEO, images[0]))
        height, width, layers = frame.shape
        
        video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*"MJPG"), 1, (width,height))
        
        for image in images:
            video.write(cv2.imread(os.path.join(self.PATH_TO_VIDEO, image)))
        
        cv2.destroyAllWindows()
        video.release()
        
        return
    
    def print_lat_long(self):
        """ Function that prints the latitude and longitude
        """
        
        for data_point in self.sorted_files:
            lat = data_point[2]
            long = data_point[3]
            
            print (lat,',',long)
            
        

if __name__ == '__main__':
    PATH_TO_VIDEO = os.path.join('..','..','data','video')
    
    EPSG = PlotEPSGcoordinates(PATH_TO_VIDEO)
    sorted_files = EPSG.sorted_files
    #EPSG.make_video()
    EPSG.print_lat_long()
    

    

    
