/**
 * particle_filter.cpp
 *
 * Created on: Dec 12, 2016
 * Author: Tiffany Huang
 */

#include "particle_filter.h"

#include <math.h>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "helper_functions.h"
#include "map.h"


using std::string;
using std::vector;
using std::normal_distribution;
using std::discrete_distribution;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  /**
   * TODO: Set the number of particles. Initialize all particles to 
   *   first position (based on estimates of x, y, theta and their uncertainties
   *   from GPS) and all weights to 1. 
   * TODO: Add random Gaussian noise to each particle.
   * NOTE: Consult particle_filter.h for more information about this method 
   *   (and others in this file).
   */
  num_particles = 100;  // TODO: Set the number of particles
  std::default_random_engine gen;

  normal_distribution<double> dist_x(x, std[0]);
  normal_distribution<double> dist_y(y, std[1]);
  normal_distribution<double> dist_theta(theta, std[2]);
  for (int p = 0; p < num_particles; ++p) {
    Particle particle = {
            p,
            dist_x(gen),
            dist_y(gen),
            dist_theta(gen),
            1.0,
            vector<int>(),
            vector<double>(),
            vector<double>()

    };
    particles.push_back(particle);

  }

}

void ParticleFilter::prediction(double delta_t, double std_pos[], 
                                double velocity, double yaw_rate) {
  /**
   * TODO: Add measurements to each particle and add random Gaussian noise.
   * NOTE: When adding noise you may find std::normal_distribution 
   *   and std::default_random_engine useful.
   *  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
   *  http://www.cplusplus.com/reference/random/default_random_engine/
   */
   std::default_random_engine gen;

    normal_distribution<double> dist_x;
    normal_distribution<double> dist_y;
    normal_distribution<double> dist_theta;

   double new_x, new_y, new_theta;
    for (auto particle : particles) {
        new_x = particle.x + velocity / yaw_rate * (sin(particle.theta + yaw_rate * delta_t) - sin(particle.theta));
        new_y = particle.y + velocity / yaw_rate * (cos(particle.theta) - cos(particle.theta + yaw_rate * delta_t));
        new_theta = particle.theta + yaw_rate * delta_t;
        dist_x = normal_distribution<double> (new_x, std_pos[0]);
        dist_y = normal_distribution<double> (new_y, std_pos[1]);
        dist_theta = normal_distribution<double> (new_theta, std_pos[2]);
        particle.x = dist_x(gen);
        particle.y = dist_y(gen);
        particle.theta = dist_theta(gen);
    }

}


/**
   * TODO: Find the predicted measurement represented by transformedXobservation and transformedYobservation that is
   * closest to each observed measurement and assign the observed measurement to this
   *   particular landmark.   *
   */
Map::single_landmark_s ParticleFilter::dataAssociation(vector<int> &associations,
                                     double transformedXobservation,
                                     double transformedYobservation,
                                     const vector<Map::single_landmark_s> &landmarksInRange) {

    double bestDistanceEst = dist(transformedXobservation, transformedYobservation,
                                  landmarksInRange[0].x_f, landmarksInRange[0].y_f);
    Map::single_landmark_s bestLandmark{};

    for (auto landmark : landmarksInRange) {
        double distance = dist(transformedXobservation, transformedYobservation, landmark.x_f, landmark.y_f);
        if (distance <= bestDistanceEst){
            bestDistanceEst = distance;
            bestLandmark = landmark;
        }
    }
    associations.push_back(bestLandmark.id_i);
    return bestLandmark;

}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
                                   const vector<LandmarkObs> &observations, 
                                   const Map &map_landmarks) {
  /**
   * TODO: Update the weights of each particle using a mult-variate Gaussian 
   *   distribution. You can read more about this distribution here: 
   *   https://en.wikipedia.org/wiki/Multivariate_normal_distribution
   * NOTE: The observations are given in the VEHICLE'S coordinate system. 
   *   Your particles are located according to the MAP'S coordinate system. 
   *   You will need to transform between the two systems. Keep in mind that
   *   this transformation requires both rotation AND translation (but no scaling).
   *   The following is a good resource for the theory:
   *   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
   *   and the following is a good resource for the actual equation to implement
   *   (look at equation 3.33) http://planning.cs.uiuc.edu/node99.html
   */
    weights = vector<double>();

    for (auto &particle : particles) {
        particle.sense_x = vector<double>();
        particle.sense_y = vector<double>();
        particle.associations = vector<int>();
        double transformedXobservation;
        double transformedYobservation;
        double particle_weight = 1.0;

        vector<Map::single_landmark_s> landmarksInRange = getLandmarksWithinRange(sensor_range, map_landmarks, particle);

        for (auto &observation : observations) {
            // Step 1: transform observations from car to map coordinates system
            transformedXobservation = convertCarToMapCoordinate(observation.x, observation.y, particle.theta, particle.x, "x");
            transformedYobservation = convertCarToMapCoordinate(observation.x, observation.y, particle.theta, particle.y, "y");
            particle.sense_x.push_back(transformedXobservation);
            particle.sense_y.push_back(transformedYobservation);

            //Step 2: assign each observation to the closest landmark
            Map::single_landmark_s best_landmark = dataAssociation(particle.associations, transformedXobservation,
                                                                          transformedYobservation, landmarksInRange);

            // Step 3: Calculate the weight of the particle
            particle_weight *= multivProb(std_landmark[0], std_landmark[1], transformedXobservation,
                                          transformedYobservation, best_landmark.x_f, best_landmark.y_f);
        }
        particle.weight = particle_weight;
        weights.push_back(particle_weight);
    }
}


vector<Map::single_landmark_s> ParticleFilter::getLandmarksWithinRange(double sensor_range, const Map &map_landmarks,
                                                                       Particle &particle){
    vector<Map::single_landmark_s> landmarks_in_range;
    for (auto landmark : map_landmarks.landmark_list) {
        double  distance = dist(particle.x, particle.y, landmark.x_f, landmark.y_f);
        if(distance <= sensor_range){
            landmarks_in_range.push_back(landmark);
        }
    }
    return landmarks_in_range;

}

void ParticleFilter::resample() {
  /**
   * TODO: Resample particles with replacement with probability proportional 
   *   to their weight. 
   * NOTE: You may find std::discrete_distribution helpful here.
   *   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
   */

  std::random_device rd;
  std::mt19937 gen(rd());
  discrete_distribution<int> distribution(weights.begin(), weights.end());
  vector<Particle> selected_particles;
  int random_index;
    for (int i = 0; i < num_particles; ++i) {
        random_index = distribution(gen);
        selected_particles.push_back(particles[random_index]);
    }

    particles = selected_particles;

}

void ParticleFilter::SetAssociations(Particle& particle, 
                                     const vector<int>& associations, 
                                     const vector<double>& sense_x, 
                                     const vector<double>& sense_y) {
  // particle: the particle to which assign each listed association, 
  //   and association's (x,y) world coordinates mapping
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates
  particle.associations= associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best) {
  vector<int> v = best.associations;
  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseCoord(Particle best, string coord) {
  vector<double> v;

  if (coord == "X") {
    v = best.sense_x;
  } else {
    v = best.sense_y;
  }

  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}