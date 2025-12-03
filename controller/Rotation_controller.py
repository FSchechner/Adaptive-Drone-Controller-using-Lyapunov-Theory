import numpy as np
class controller (): 
    def __init__ (self): 
        self.Kp = 0.0 
        self.Kp = 0.0 
        self.eprevp 
        self.eprevq
        self.eprevr 
        self.ep 
        self.eq
        self.er 
        self.dp
        self.dq
        self.dr
        self.dt
    
    def main (self,p,q,r,p_d,q_d,r_d): 
        self. calc_errors (p,q,r,p_d,q_d,r_d)
        self.derivatives ()
        taup, tauq, taur= self.controll ()
        return taup, tauq, taur

    def controll (self): 
        taup = self.Kp * self.ep + self.Kd * self.edp
        tauq = self.Kp * self.eq + self.Kd * self.edq
        taur = self.Kp * self.er + self.Kd * self.edr 
        return taup, tauq, taur
    
    def derivatives (self): 
        self.dp = (self.eprevp - self.ep) * self.dt
        self.dq = (self.eprevq - self.eq) * self.dt
        self.dr = (self.eprevr - self.er) * self.dt
    
    def calc_errors (self,p,q,r,p_d,q_d,r_d): 
        self.eprevp = self.ep
        self.eprevq = self.eq
        self.eprevr = self.er

        self.ep = p_d - p
        self.eq = q_d - q
        self.er = r_d - r
    
