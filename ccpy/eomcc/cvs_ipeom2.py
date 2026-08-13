"""Ionization Potential Equation-of-Motion Coupled-Cluster
Method with 1h and 2h-1p Excitations on top of CCSD [IP-EOMCCSD(2h-1p)]"""
import numpy as np
from ccpy.lib.core import cc_loops2

# R.a -> (noa) -> (i)
# R.aa -> (noa,nua,noa) -> (ibj)
# R.ab -> (noa,nub,nob) -> (ib~j~)

def update(R, omega, H, RHF_symmetry, system):

    R.a, R.aa, R.ab = cc_loops2.update_r_2h1p(
        R.a,
        R.aa,
        R.ab,
        omega,
        H.a.oo,
        H.a.vv,
        H.b.oo,
        H.b.vv,
        0.0,
    )
    return R

def HR(dR, R, T, H, flag_RHF, system):
    core = 2

    Ra   = {"c" : {}}
    Raa  = {"c" : {}, "cc" : {}, "cv" : {}}
    Rab  = {"c" : {}}

    Ra["c"]      = R.a[:core]

    Raa["c"]     = R.aa[:core, :, :]
    Raa["cc"]    = R.aa[:core, :, :core]
    Raa["cv"]    = R.aa[:core, :, core:]

    Rab["c"]     = R.ab[:core, :, :]

    I = (
        - np.einsum("mnef,mfn->e", H.aa.oovv[:core, core:, :, :], Raa["cv"], optimize=True) 
        - 0.5 * np.einsum("mnef,mfn->e", H.aa.oovv[:core, :core, :, :], Raa["cc"], optimize=True) 
        - np.einsum("mnef,mfn->e", H.ab.oovv[:core, :, :, :], Rab["c"], optimize=True) 
        )
    # update R1
    dR.a = build_HR_1A(R, Ra, Raa, Rab, T, H, core)
    # update R2
    dR.aa = build_HR_2A(R, Ra, Raa, Rab, T, H, I, core)
    dR.ab = build_HR_2B(R, Ra, Raa, Rab, T, H, I, core)

    return dR.flatten()

def build_HR_1A(R, Ra, Raa, Rab, T, H, core):
    """Calculate the projection <I|[ (H_N e^(T1+T2))_C*(R1h+R2h1p) ]_C|0>."""
    X1A = np.zeros_like(R.a)

    X1A[:core] -= np.einsum("mi,m->i", H.a.oo[:core, :core], Ra["c"], optimize=True)
    X1A[:core] -= np.einsum("mnif,mfn->i", H.aa.ooov[:core, core:, :core, :], Raa["cv"], optimize=True)
    X1A[:core] -= 0.5 * np.einsum("mnif,mfn->i", H.aa.ooov[:core, :core, :core, :], Raa["cc"], optimize=True)
    X1A[:core] -= np.einsum("mnif,mfn->i", H.ab.ooov[:core, :, :core, :], Rab["c"], optimize=True)
    X1A[:core] += np.einsum("me,iem->i", H.a.ov, Raa["c"], optimize=True)
    X1A[:core] += np.einsum("me,iem->i", H.b.ov, Rab["c"], optimize=True)

    return X1A

def build_HR_2A(R, Ra, Raa, Rab, T, H, I, core):
    """Calculate the projection <Ijb|[ (H_N e^(T1+T2))_C*(R1h+R2h1p) ]_C|0>."""
    X2A = np.zeros_like(R.aa)

    X2A[:core, :, :core] -= 0.5 * np.einsum("bmji,m->ibj", H.aa.vooo[:, :core, :core, :core], Ra["c"], optimize=True)
    X2A[:core, :, core:] -= np.einsum("bmji,m->ibj", H.aa.vooo[:, :core, core:, :core], Ra["c"], optimize=True)
    X2A[:core, :, :core] += 0.5 * np.einsum("be,iej->ibj", H.a.vv, Raa["cc"], optimize=True)
    X2A[:core, :, core:] += np.einsum("be,iej->ibj", H.a.vv, Raa["cv"], optimize=True)
    X2A[:core, :, :core] += np.einsum("mi,jbm->ibj", H.a.oo[:, :core], Raa["c"],  optimize=True) # change
    X2A[core:, :, :core] -= np.einsum("mi,mbj->ibj", H.a.oo[:core, core:], Raa["cc"], optimize=True)
    X2A[:core, :, core:] -= np.einsum("mi,mbj->ibj", H.a.oo[:core, :core], Raa["cv"], optimize=True)
    X2A[:core, :, :core] += np.einsum("bmje,iem->ibj", H.aa.voov[:, :, :core, :], Raa["c"], optimize=True)
    X2A[core:, :, :core] -= np.einsum("bmje,mei->ibj", H.aa.voov[:, :core, :core, :], Raa["cv"], optimize=True) # change
    X2A[:core, :, core:] += np.einsum("bmje,iem->ibj", H.aa.voov[:, :core, core:, :], Raa["cc"], optimize=True)
    X2A[:core, :, :core] += np.einsum("bmje,iem->ibj", H.ab.voov[:, :, :core, :], Rab["c"], optimize=True)
    X2A[:core, :, core:] += 0.5  * np.einsum("mnij,mbn->ibj", H.aa.oooo[:core, :core, :core, core:], Raa["cc"], optimize=True)
    X2A[:core, :, core:] += np.einsum("mnij,mbn->ibj", H.aa.oooo[:core, core:, :core, core:], Raa["cv"], optimize=True)
    X2A[:core, :, :core] += 0.25 * np.einsum("mnij,mbn->ibj", H.aa.oooo[:core, :core, :core, :core], Raa["cc"], optimize=True)
    X2A[:core, :, :core] += 0.5 * np.einsum("mnij,mbn->ibj", H.aa.oooo[:core, core:, :core, :core], Raa["cv"], optimize=True)
    X2A[:core, :, :core] += 0.5 * np.einsum("e,ebij->ibj", I, T.aa[:, :, :core, :core], optimize=True)
    X2A[:core, :, core:] += np.einsum("e,ebij->ibj", I, T.aa[:, :, :core, core:], optimize=True)

    X2A -= np.transpose(X2A, (2, 1, 0))

    tmp  = - np.einsum("mj,ibm->ibj", H.a.oo[core:, core:], Raa["cv"], optimize=True) # change
    tmp += np.einsum("bmje,iem->ibj", H.aa.voov[:, core:, core:, :], Raa["cv"], optimize=True)
    tmp += np.einsum("bmje,iem->ibj", H.ab.voov[:, :, core:, :], Rab["c"], optimize=True)

    X2A[:core, :, core:] += tmp

    X2A[core:, :, :core] -= tmp.transpose(2, 1, 0)

    return X2A

def build_HR_2B(R, Ra, Raa, Rab, T, H, I, core):
    """Calculate the projection <ij~b~|[ (H_N e^(T1+T2))_C*(R1h+R2h1p) ]_C|0>."""
    X2B = np.zeros_like(R.ab)

    X2B[:core, :, :] -= np.einsum("mbij,m->ibj", H.ab.ovoo[:core, :, :core, :], Ra["c"], optimize=True)
    X2B[:core, :, :] -= np.einsum("mi,mbj->ibj", H.a.oo[:core, :core], Rab["c"], optimize=True)
    X2B[:core, :, :] -= np.einsum("mj,ibm->ibj", H.b.oo, Rab["c"], optimize=True)
    X2B[:core, :, :] += np.einsum("be,iej->ibj", H.b.vv, Rab["c"], optimize=True)
    X2B[:core, :, :] += np.einsum("mnij,mbn->ibj", H.ab.oooo[:core, :, :core, :], Rab["c"], optimize=True)
    X2B[:core, :, :] += np.einsum("mbej,iem->ibj", H.ab.ovvo[:core, :, :, :], Raa["cc"], optimize=True)
    X2B[:core, :, :] += np.einsum("mbej,iem->ibj", H.ab.ovvo[core:, :, :, :], Raa["cv"], optimize=True)
    X2B[:core, :, :] += np.einsum("bmje,iem->ibj", H.bb.voov, Rab["c"], optimize=True)
    X2B[:core, :, :] -= np.einsum("mbie,mej->ibj", H.ab.ovov[:core, :, :core, :], Rab["c"], optimize=True)
    X2B[:core, :, :] += np.einsum("e,ebij->ibj", I, T.ab[:, :, :core, :], optimize=True)

    return X2B
