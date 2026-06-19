# BASEBALL PITCH MECHANICS ANALYZER

This is a program made with python to analyze pitching mechanics with just a SMARTPHONE. 
This can automatically measure **Hip Shoulder Separation, Stride Length, Release Extension Length**

## Why did I made this program?

I was in a long slump of increasing my velocity in my high school varsity baseball team. I researched the primary factor that divides amatuer pitcher and an elite pitcher throwing 90+ miles. It was all about utitlizing ur body's **kinetic chain** efficiently. The intricator to measure the efficiency of the kinetic chain was measuring hip shoulder separation, stride length, and release extension length. This can be completed in numerous ways such as comparing each videos, using motion capture suits, or multiple slow motion cameras in different angle. Since these approaches are impossible in a high school baseball field, I attempted to create a program that can conveniently measure all three elements just with a smartphone. 

## Errors/Trials

## 1. Inconsistency on measuring the pitcher
While analyzing the video behind the catcher, my program often gave results that my hip shoulder seperation was over 80 degreees which is impossible for a human being to have. This was because my program captured the person who appears to be the biggest as the pitcher, setting the catcher as the pitcher. 
**Solve**: I added a logic to automatically set up the smallest appearing person as the pithcer. This worked since the pitcher is always further from the camera than the cather is. 

## 2. 