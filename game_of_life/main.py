#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    Alien game demo (a.k.a. game of life)

    # TODO test case: read map, count number of cities, write map, read map, count number of cities (assert)
    # TODO test case: world_map_small contains 28 lines
    # TODO test case: world_map_medium contains 6763 lines
    # TODO test case: create world full of not-connected cities, artificially place two aliens everywhere, check if all cities are destroy

"""

# built-in libs
import argparse
import logging
import os
import random
import string
import sys
# standard libs


__author__ = "Eduard Feicho"
__copyright__ = "Copyright 2015"


logger = logging.getLogger(__name__)




class AlienApp():

    def __init__(self):
        # Setup an empty args object, which contains the same variables as the command line argument parser would produce.
        # This way we can either run the python script from command line or alternatively instantiate the class as a module
        # and adjust settings using the args object
        self.args           = lambda: None
        self.args.logfile   = None
        self.args.verbose   = 0
        self.args.debug     = False
        self.args.numAliens = 1

        self.args.input  = 'world_map_small.txt'
        self.args.output = None

        # initialise random number generator
        random.seed()



    def run(self):
        """
            Main entry point of the game
        """
        self.readMap()
        self.createAliens()
        self.runLoop()
        self.writeMap()



    def readMap(self):
        """
            Read the world map from an input file
        """
        self.world = World.readMap(self.args.input)
        logger.info('Loaded world of {} cities'.format(len(self.world.cities)))



    def writeMap(self):
        """
            Write the world map to an output file
        """
        if not self.args.output:
            return
        self.world.writeMap(self.args.output)
        logger.info('Written world of {} cities'.format(len(self.world.cities)))



    def createAliens(self):
        """
            Create aliens initially

            Assumptions:
            - Assume not two aliens will initially occupy the same city and immediately destroy it.
            - Assume not more aliens can initially be on the map than the number of cities that exist.
        """

        # get the initial set of available (unoccupied) cities
        cityKeys = self.world.cities.keys()
        numCities = len(cityKeys)

        # Assume not two aliens will initially occupy the same city and immediately destroy it.
        # Also, assume not more aliens can initially be on the map than cities exist.
        N = min(self.args.numAliens, numCities)

        logger.debug(' -- Creating {} aliens'.format(N))

        self.aliens = []
        for i in range(N):

            # Update name
            alien = Alien(name = str(i))
            self.aliens.append(alien)

            # Chose city randomly (and remove) from the list of unoccupied cities
            rand = random.randint(0, len(cityKeys)-1)
            cityName = cityKeys.pop(rand)
            alien.moveTo(cityName)

            logger.debug('Created alien {} at {}'.format(alien.name, alien.city))


    def runLoop(self):
        """
            The run loop of the game (terminates when game is over or after 10000 steps)
        """

        aliensAlive = len(self.aliens)
        running = True

        # count the number of steps/frames and terminate the game after 10000 steps
        steps = 0

        self.printInfo()
        while running and steps < 10000:

            # step update (move aliens)
            self.moveAliens()

            # check if two aliens in same city
            self.fightAliens()

            steps += 1

            self.printInfo()

            # terminate the game when
            # - aliens cannot move anymore (all are trapped or dead)
            # - all cities are destroyed
            if self.countAliensMoving() <= 1 or self.countCitiesAlive() == 0:
                running = False

        logger.info('Game over')



    def printInfo(self):
        countCities         = self.countCities()
        countCitiesAlive    = self.countCitiesAlive()
        countAliens         = self.countAliens()
        countAliensAlive    = self.countAliensAlive()
        countAliensTrapped  = self.countAliensTrapped()
        
        logger.debug(' #cities {}/{} alive, #aliens {}/{} alive ({} trapped)'.format(countCitiesAlive, countCities, countAliensAlive, countAliens, countAliensTrapped))


    def countCities(self):
        return len(self.world.cities)

    def countCitiesAlive(self):
        count = 0
        #print self.world.cities.iteritems():
        for key,city in self.world.cities.iteritems():
            if not city.destroyed:
                count += 1
        return count

    def countAliens(self):
        return len(self.aliens)

    def countAliensMoving(self):
        return self.countAliensAlive() - self.countAliensTrapped()

    def countAliensAlive(self):
        count = 0
        for alien in self.aliens:
            if alien.alive:
                count += 1
        return count

    def countAliensTrapped(self):
        count = 0
        for alien in self.aliens:
            if alien.trapped:
                count += 1
        return count


    def moveAliens(self):
        """
            These aliens [...] wander around randomly, following links.
            Each iteration, the aliens can travel in any of the directions leading out of a city.
        """
        for alien in self.aliens:

            # dead aliens can't move anyway
            if not alien.alive:
            	continue

            # trapped aliens can't move anyway
            if alien.trapped:
                # logger.debug(' Alien {} is trapped in {}'.format(alien.name, alien.city))
                continue
            
            currentCity = self.world.cities[alien.city]
            availableDirections = []
            
            for (direction, value) in currentCity.links.iteritems():
                if value == None:
                    continue
                availableDirections.append(direction)

            if len(availableDirections) == 0:
                # this alien is trapped
                alien.trapped = True
                continue

            rand = random.randint(0, len(availableDirections)-1)
            newCity = currentCity.links[availableDirections[rand]]

            logger.debug(' Alien {} moves from {} to {}'.format(alien.name, alien.city, newCity))
            alien.moveTo(newCity)




    def fightAliens(self):
        """
            When two aliens end up in the same place, they fight, and in the process kill
            each other and destroy the city. When a city is destroyed, it is removed from
            the map, and so are any roads that lead into or out of it.

            Assumptions: When there are more than two aliens in the same city, they all die.
        """

        for i in range(len(self.aliens)):
            alienA = self.aliens[i]

            # TODO: these could be filtered out completely instead of being kept (think memory/speed consumption)
            if not alienA.alive:
                continue

            for j in range(i+1, len(self.aliens)):
                alienB = self.aliens[j]
                if not alienB.alive:
                    continue

                if alienA.city == alienB.city:
                    # fight
                    alienA.kill()
                    alienB.kill()
                    # destroy city (and roads leading to it)
                    self.world.destroyCity(alienA.city)

                    logger.info('{} has been destroyed by alien {} and alien {}!'.format(alienA.city, alienA.name, alienB.name))



class Alien():

    def __init__(self, name):
        self.name = name
        self.trapped = False
        self.alive = True

    def moveTo(self, cityName):
        self.city = cityName

    def kill(self):
        self.alive = False

    def trapped(self):
        self.trapped = True



class City():

    def __init__(self, name, north=None, east=None, south=None, west=None):
        self.name  = name
        self.links = dict()
        self.links['north'] = north
        self.links['east']  = east
        self.links['south'] = south
        self.links['west']  = west
        self.destroyed = False

    def destroy(self):
        self.links['north'] = None
        self.links['east']  = None
        self.links['south'] = None
        self.links['west']  = None
        self.destroyed = True



class World():

    def __init__(self, cityMapping=[]):
        self.cities = dict(cityMapping)
        pass

    def addCity(self, city):
        self.cities[city.name] = city

    def destroyCity(self, cityName):
        """
            When a city is destroyed, it is removed from the map, and so are any roads that lead into or out of it.
        """
        self.cities[cityName].destroy()

        # loop through all cities in the world and remove roads to the destroyed city
        for (key,city) in self.cities.iteritems():
            for (linkKey,linkValue) in city.links.iteritems():
                if linkValue == cityName:
                    city.links[linkKey] = None


    @staticmethod
    def readMap(filename):
        """
            Read world from file.
            Assume each line contains a unique city whose name does not contain spaces.
            Within each line, the city name and it's outgoing roads are separated by spaces.
            Each outgoing road is defined as a key=value pair, where
            - key is one out of {north,east,south,west}
            - value is a city name
        """

        world = World(cityMapping=[])

        try:

            lines = [line.strip() for line in open(filename, 'r')]    # read line-by-line and trim leading/trailing spaces/newlines
            lines = filter(None, lines)                               # filter out empty lines

            for i in range(len(lines)):
                line = lines[i]

                # Assume city name and link descriptions don't contain spaces and all city links can be separated/tokenized by a single space character
                tokens = line.split(' ')

                # Assume city name is at very beginning of line and the name ends when the first space occurs
                cityName = tokens.pop(0)

                # Create city object
                city = City(cityName)

                # Loop through the city link descriptions
                for link in tokens:

                    # Assume each link description is of the form 'DIRECTION=CITYNAME' where DIRECTION is in {north,east,south,west} and CITYNAME is any string
                    direction, linkName = link.split('=')
                    # update the city links dictionary
                    city.links[direction] = linkName

                world.addCity(city)

        except Exception as e:
            logger.error(e)
            logger.error('Error reading file {}'.format(filename))

        return world


    def writeMap(self, filename):
        """
            ...
        """
        try:
            with open(filename, 'w') as out:
                for (name, city) in self.cities.iteritems():
                    line = '{}'.format(name)
                    for (direction, linkName) in city.links.iteritems():
                    	if linkName == None:
                    		continue
                        line += ' {}={}'.format(direction, linkName)

                    logger.debug(line)
                    out.write(line + '\n')

        except Exception as e:
            logger.error(e)
            logger.error('Error writing file {}'.format(filename))






def main():

    # initialize the app (contains default values for some command line arguments)
    app = AlienApp()

    # setup command line argument parser
    parser = argparse.ArgumentParser(description="Alien invasion demo")
    parser.add_argument("input",                metavar="MAP",          help="Path to map file that is written at end of execution")
    parser.add_argument("-n", "--aliens",       dest="numAliens",       default=app.args.numAliens,  type=int,  help="Number of aliens that are created initially")
    parser.add_argument("-o", "--output",       dest="output",          default=app.args.output,                help="Path to map file that is written at end of execution")
    parser.add_argument("-l", "--logfile",      dest="logfile",         default=None,                           help="Path to logfile")
    parser.add_argument("-v", "--verbose",      dest="verbose",         default=1,      action="count",         help="verbosity level (use -vv to increase more)")
    parser.add_argument("-q", "--quiet",        dest="quiet",           default=False,  action="store_true",    help="don't print status messages to stdout")
    parser.add_argument("--test",               dest="test",            default=False,  action="store_true",    help="run unit tests (TODO)")
    parser.add_argument("--debug",              dest="debug",           default=False,  action="store_true",    help="enable debugging mode")

    # parse command line arguments
    args = parser.parse_args()
    if args.quiet: args.verbose = 0
    if args.debug: args.verbose = 2

    # setup logging from command line arguments
    setupLogging(args, logger)

    # use command line arguments to specify the configuration of the app
    app.args = args

    # the main entry point of the app
    app.run()




# setup logging from given command line arguments
def setupLogging(args, logger, basicFormat='%(levelname)s:\t%(message)s', fileFormat='%(asctime)s\t%(levelname)s:\t%(message)s', loglevel=None):
    """
        
    """

    # if logging level is not explcitly specified, chose based on verbosity level specified on command line
    if loglevel is None:
        loglevel = logging.WARNING                            # default to log level WARNING
        if args.verbose >= 1: loglevel = logging.INFO         # specify for example -v or --verbose
        if args.verbose >= 2: loglevel = logging.DEBUG         # specify for example -vv

    logging.basicConfig(format=basicFormat, level=loglevel)

    # write to file, if -l/--logfile specified on command line
    if args.logfile is not None:
        handler = logging.FileHandler(args.logfile)
        formatter = logging.Formatter(fileFormat)
        handler.setLevel(loglevel)
        handler.setFormatter(formatter)
        logger.addHandler(handler)



if __name__ == "__main__":
    main()
