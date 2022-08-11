# PySanket
Developing a solution for Gesture enabled commands for operating Laptops/PCs for frequently used operations on daily basis.

- For any user, authenticated by face recognition, few gestures could be defined for frequently used tasks- save, exit, print, screen-lock, screen unlock, system shut down, system restart. 
- Save, print and exit operations are context sensitive meaning that it is applicable for current application. 
- For example, if word document is open and the gesture for save is done then the document will save, if print gesture is done then printer dialog will open etc. 
- Similarly, a gesture could be defined for close/exit which will close the current application. 
- If no application is opened, then it will work as system shut down. It is like Alt+F4 key press functionality on windows PC.

### This Project was presented in Smart India Hackathon 2022 with problem code :- NR1167

This repository contains the following contents.

- Sample program
- Hand sign recognition model(TFLite)
- Learning data for hand sign recognition and notebook for learning

## Requirements
- mediapipe 0.8.1
- OpenCV 3.4.2 or Later
- Tensorflow 2.3.0 or Later
- scikit-learn 0.23.2 or Later (Only if you want to display the confusion matrix)
- matplotlib 3.3.2 or Later (Only if you want to display the confusion matrix)

## Demo
Here's how to run the demo using your webcam.
```
python app.py
```

### Here's video of working of project 
[![Smart India Hackathon 2022 || NR1167 || Team Pythonic || Project PySanket || Nirma University](https://youtube-md.vercel.app/OXiEkLhiUA0/640/360)](https://www.youtube.com/watch?v=OXiEkLhiUA0)

## Contributing

1.  Fork the repository
2.  Do the desired changes (add/delete/modify)
3.  Make a pull request

## Contributors (SIH Team)
1. [Bhargav Modha](https://github.com/bhargav-modha/)
2. [Akshay Patel](https://github.com/akshaypatel67/)
3. [Harsh Maniar ](https://github.com/HarshManiar1804)
4. [Rajvi Desai](https://github.com/CuriousRajvi)
5. Kishan Gondaliya
6. Disha Ambade
