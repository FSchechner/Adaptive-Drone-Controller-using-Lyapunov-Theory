import numpy as np

class controller (): 
    def __init__ (self): 
        self.dt

        self.Kp = 0.0
        self.Kd = 0.0 
      
        self.dex = 0.0 
        self.dey = 0.0 
        self.dez = 0.0 
       
        self.prevex = 0.0 
        self.prevey = 0.0 
        self.prevez = 0.0 
    
        self.ex = 0.0
        self.ey = 0.0
        self.ez = 0.0

    def main (self,x_d,y_d,z_d,x,y,z): 
        self.calc_derivatives ()
        self.calc_errors(x_d,y_d,z_d,x,y,z)
        u = self.calc_control(self)
        return u

    def calc_control(self): 
        ux = self.Kp * self.ex + self.Kd * self.dex
        uy = self.Kp * self.ey + self.Kd * self.dey
        uz = self.Kp * self.ez + self.Kd * self.dez
        return np.array([ux,uy,uz])

    def calc_errors (self,x_d,y_d,z_d,x,y,z): 
        self.prevex = self.ex
        self.prevey = self.ey 
        self.prevez = self.ez 

        self.ex = x_d - x
        self.ey = y_d - y
        self.ez = z_d - z

    def calc_derivatives (self):
        self.dx = (self.prevx - self.x) * self.dt
        self.dy = (self.prevy - self.y) * self.dt
        self.dz = (self.prevz - self.z) * self.dt
