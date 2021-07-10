# ROS-hackathon  

## Our Stack   
The stack made consists of a path planner (`path_planner.py`) which uses `RRT*` algorithm to find the path as a list of points and `Dijkstra` algorithm to optimize the path which is then used by the PID node (`pid.py`) to
move the omnibase bot in gazebo by publishing to the /cmd_vel topic. Bothe the planner and the pid node are present in the `~omnibase_control/scripts` folder
  
## Use
    
In the catkin workspace:  
- run `catkin_make`  
- run `source devel/setup.bash`  
- run `roslaunch omnibase_gazebo hackathon_omnibase.launch`  
- cd to `~/omnibase/omnibase_control/scripts` and then run the `pid.py` script  
  
This would prompt you to enter the final coordinates for the bot, enter the correct coordinates and watch the bot navigate to it on gazebo  
  
## Note  
  
Due to time constraints the path_planner.py is not a ROS node but just a python script which is imported into the PID node and used to find the path.
Further the obstacle detector node too has been replaced by hardcoding the obstacles into the path_planner.py script.  
  
## Contributors:  
  
- Siddh Gosar  
- Arnav Saraf  
- Soham Chitnis
