# This code helps compute the basin of attraction of an extended game with gifting actions.
# The risk and gifting values can be set in the main function (lines 121-122)
# WARNING: This code requires a lot of computation and may not be feasible for personal computers.
#          On a cloud computing system, it took more than 2 hours to run for each risk-gift pair.
#          One can reduce the precision by decreasing the 11's and 101's in the main function.
# The output of the code is a numpy array (saved) that consists of boolean values of whether the
# corresponding point of policy pair reaches the prosocial equilibrium or not.

import numpy as np

def x2s(x):
    # converts the relative differences w.r.t. the forage parameters into action probabilities
    # Input: 6xN matrix where rows correspond to x_1 - x_2, x_3 - x_2, x_4 - x_2 (and the same for y),
    #        and N is the batch size for more efficient computation.
    # Output: 8xN matrix where rows correspond to the action probabilities.
    
    s = np.exp(x)
    s1 = np.sum(s[:3],axis=0)+1
    s2 = np.sum(s[3:],axis=0)+1
    return np.array([s[0]/s1, 1/s1, s[1]/s1, s[2]/s1, s[3]/s2, 1/s2, s[4]/s2, s[5]/s2])


def prosocial_equ(x, r=-6, gamm=2):
    # computes whether a given point in the dynamical system (pairs of policy parameters) reaches the
    # prosocial equilibrium or not for the given risk (r) and gifting (gamm) values.
    # Input: x is a 6xN matrix where rows correspond to x_1 - x_2, x_3 - x_2, x_4 - x_2 (and the same for y),
    #        and N is the batch size for more efficient computation.
    #        r is the risk value (hunt-alone payoff)
    #        gamm is the gifting amount
    # Output: N-vector of booleans where each entry is True if the system goes to the prosocial equilibrium
    #         and False otherwise.
    
    # Define the probabilities in the equilibria
    p1 = np.array([1., 0., 0., 0., 1., 0., 0., 0.]).reshape(-1,1)
    p2 = np.array([0., 1., 0., 0., 0., 1., 0., 0.]).reshape(-1,1)
    
    # We will update the systems (there are N of them) until all of them converge
    condition = False
    while not condition:
    
        # Get action probabilitites
        ps1, pm1, psg1, pmg1, ps2, pm2, psg2, pmg2 = x2s(x)

        # Compute the gradients       
        E11 = ps1 * (pm1 + psg1 + pmg1)
        E12 = - ps1 * pm1
        E13 = - ps1 * psg1
        E14 = - ps1 * pmg1
                   
        E21 = - pm1 * ps1
        E22 = pm1 * (ps1 + psg1 + pmg1)
        E23 = - pm1 * psg1
        E24 = - pm1 * pmg1
                   
        E31 = - psg1 * ps1
        E32 = - psg1 * pm1
        E33 = psg1 * (ps1 + pm1 + pmg1)
        E34 = - psg1 * pmg1
                   
        E41 = - pmg1 * ps1
        E42 = - pmg1 * pm1
        E43 = - pmg1 * psg1
        E44 = pmg1 * (ps1 + pm1 + psg1)
                   
        E55 = ps2 * (pm2 + psg2 + pmg2)
        E56 = - ps2 * pm2
        E57 = - ps2 * psg2
        E58 = - ps2 * pmg2
                   
        E65 = - pm2 * ps2
        E66 = pm2 * (ps2 + psg2 + pmg2)
        E67 = - pm2 * psg2
        E68 = - pm2 * pmg2
                   
        E75 = - psg2 * ps2
        E76 = - psg2 * pm2
        E77 = psg2 * (ps2 + pm2 + pmg2)
        E78 = - psg2 * pmg2
                   
        E85 = - pmg2 * ps2
        E86 = - pmg2 * pm2
        E87 = - pmg2 * psg2
        E88 = pmg2 * (ps2 + pm2 + psg2)

        # Instead of updating all 8 parameters and then shifting everything to make Forage parameters 0, we can do this shift on the gradients
        # Therefore, the gradients vector will be 6 dimensional
        x_dot = np.zeros((6,x.shape[1]), dtype=float)
        
        # And we will shift the 6 gradients w.r.t. to the gradients of the forage parameters:
        x_dot_pm1 = (E12*(2*ps2 + r*pm2 + (2+gamm)*psg2 + (r+gamm)*pmg2) + E22*(1*ps2 + 1*pm2 + (1+gamm)*psg2 + (1+gamm)*pmg2) + E32*((2-gamm)*ps2 + (r-gamm)*pm2 + 2*psg2 + r*pmg2) + E42*((1-gamm)*ps2 + (1-gamm)*pm2 + 1*psg2 + 1*pmg2))
        x_dot_pm2 = (E56*(2*ps1 + r*pm1 + (2+gamm)*psg1 + (r+gamm)*pmg1) + E66*(1*ps1 + 1*pm1 + (1+gamm)*psg1 + (1+gamm)*pmg1) + E76*((2-gamm)*ps1 + (r-gamm)*pm1 + 2*psg1 + r*pmg1) + E86*((1-gamm)*ps1 + (1-gamm)*pm1 + 1*psg1 + 1*pmg1))

        # Finalize the gradient computation for the 6 parameters
        x_dot[0] = (E11*(2*ps2 + r*pm2 + (2+gamm)*psg2 + (r+gamm)*pmg2) + E21*(1*ps2 + 1*pm2 + (1+gamm)*psg2 + (1+gamm)*pmg2) + E31*((2-gamm)*ps2 + (r-gamm)*pm2 + 2*psg2 + r*pmg2) + E41*((1-gamm)*ps2 + (1-gamm)*pm2 + 1*psg2 + 1*pmg2))
        x_dot[1] = (E13*(2*ps2 + r*pm2 + (2+gamm)*psg2 + (r+gamm)*pmg2) + E23*(1*ps2 + 1*pm2 + (1+gamm)*psg2 + (1+gamm)*pmg2) + E33*((2-gamm)*ps2 + (r-gamm)*pm2 + 2*psg2 + r*pmg2) + E43*((1-gamm)*ps2 + (1-gamm)*pm2 + 1*psg2 + 1*pmg2))
        x_dot[2] = (E14*(2*ps2 + r*pm2 + (2+gamm)*psg2 + (r+gamm)*pmg2) + E24*(1*ps2 + 1*pm2 + (1+gamm)*psg2 + (1+gamm)*pmg2) + E34*((2-gamm)*ps2 + (r-gamm)*pm2 + 2*psg2 + r*pmg2) + E44*((1-gamm)*ps2 + (1-gamm)*pm2 + 1*psg2 + 1*pmg2))

        x_dot[3] = (E55*(2*ps1 + r*pm1 + (2+gamm)*psg1 + (r+gamm)*pmg1) + E65*(1*ps1 + 1*pm1 + (1+gamm)*psg1 + (1+gamm)*pmg1) + E75*((2-gamm)*ps1 + (r-gamm)*pm1 + 2*psg1 + r*pmg1) + E85*((1-gamm)*ps1 + (1-gamm)*pm1 + 1*psg1 + 1*pmg1))
        x_dot[4] = (E57*(2*ps1 + r*pm1 + (2+gamm)*psg1 + (r+gamm)*pmg1) + E67*(1*ps1 + 1*pm1 + (1+gamm)*psg1 + (1+gamm)*pmg1) + E77*((2-gamm)*ps1 + (r-gamm)*pm1 + 2*psg1 + r*pmg1) + E87*((1-gamm)*ps1 + (1-gamm)*pm1 + 1*psg1 + 1*pmg1))
        x_dot[5] = (E58*(2*ps1 + r*pm1 + (2+gamm)*psg1 + (r+gamm)*pmg1) + E68*(1*ps1 + 1*pm1 + (1+gamm)*psg1 + (1+gamm)*pmg1) + E78*((2-gamm)*ps1 + (r-gamm)*pm1 + 2*psg1 + r*pmg1) + E88*((1-gamm)*ps1 + (1-gamm)*pm1 + 1*psg1 + 1*pmg1))

        # Do the shift
        x_dot[:3] = x_dot[:3] - x_dot_pm1
        x_dot[3:] = x_dot[3:] - x_dot_pm2

        # Update the parameters (because we are computing the basin of attraction numerically)
        x += x_dot
        
        # Convert the parameters into probabilities
        s = x2s(x)
        
        # Check if it converged to either of the equilibria
        c1 = np.linalg.norm(s-p1,axis=0) < 0.1
        c2 = np.linalg.norm(s-p2,axis=0) < 0.1
        condition = np.all(np.logical_or(c1, c2))
        
    return c1 # because p1 is the prosocial equilibrium
    

if __name__ == '__main__':
    r = -6
    gamm = 10

    # These two arrays will hold the results and whether we processed the corresponding point yet
    results = np.full((101, 11, 11, 101, 11, 11), False, dtype=bool)
    processed = np.full((101, 11, 11, 101, 11, 11), False, dtype=bool)
    
    # These six vectors correspond to the policy parameters relative to the Forage parameters, e.g., avec is for x_1 - x_2.
    avec = np.linspace(-3,3,101)
    bvec = np.linspace(-3,3,11)
    cvec = np.linspace(-3,3,11)
    dvec = np.linspace(-3,3,101)
    evec = np.linspace(-3,3,11)
    fvec = np.linspace(-3,3,11)

    for bid in range(len(bvec)):
        b = bvec[bid]
        for cid in range(len(cvec)):
            c = cvec[cid]
            for did in range(len(dvec)):
                d = dvec[did]
                for eid in range(len(evec)):
                    e = evec[eid]
                    for fid in range(len(fvec)):
                        f = fvec[fid]
                        if not np.all(processed[:,bid,cid,did,eid,fid]):
                            # We process matrices instead of vectors for the x_1 dimension (corresponding to avec) to save some computation time
                            x = np.zeros((6,101), dtype=float)
                            x[1:] = np.reshape([b,c,d,e,f],(5,1))
                            x[0] = avec.copy()
                            
                            # Compute where the current points reach and store them
                            res = prosocial_equ(x, r, gamm)
                            results[:,bid,cid,did,eid,fid] = res
                            processed[:,bid,cid,did,eid,fid] = True
                            
                            # Exploit of symmetry of the game to save half of the computation time:
                            results[did,eid,fid,:,bid,cid] = res
                            processed[did,eid,fid,:,bid,cid] = True
    
    # Save the results
    np.save('r' + str(r) + '_gamma' + str(gamm) + '.npy', results)