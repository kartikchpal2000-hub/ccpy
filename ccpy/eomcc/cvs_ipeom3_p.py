"""Ionization Potential Equation-of-Motion Coupled-Cluster
Method with 1h, 2h-1p, and an Arbitrary List of 3h-2p Excitations
on top of CCSD [IP-EOMCCSD(P)]"""
import numpy as np
from ccpy.lib.core import cvs_ipeom3_p_loops
from ccpy.eomcc.cvs_ipeom3_intermediates import get_cvs_ipeom3_p_intermediates

def update(R, omega, H, RHF_symmetry, system, r3_excitations):
    R.a, R.aa, R.ab, R.aaa, R.aab, R.abb = ipeom3_p_loops.update_r(
        R.a,
        R.aa,
        R.ab,
        R.aaa,
        r3_excitations["aaa"],
        R.aab,
        r3_excitations["aab"],
        R.abb,
        r3_excitations["abb"],
        omega,
        H.a.oo,
        H.a.vv,
        H.b.oo,
        H.b.vv,
    )
    return R

def HR(dR, R, T, H, flag_RHF, system, t3_excitations, r3_excitations):

    core = 1

    Ra   = {"c" : {}}
    Raa  = {"c" : {}, "cc" : {}, "cv" : {}}
    Rab  = {"c" : {}}

    Ra["c"]      = R.a[:core]

    Raa["c"]     = R.aa[:core, :, :]
    Raa["cc"]    = R.aa[:core, :, :core]
    Raa["cv"]    = R.aa[:core, :, core:]

    Rab["c"]     = R.ab[:core, :, :]

    # determine whether r3 updates should be done. Stupid compatibility with
    # empty sections r3_excitations
    do_r3 = {"aaa": True, "aab": True, "abb": True}
    if np.array_equal(r3_excitations["aaa"][0, :], np.array([1., 1., 1., 1., 1.])):
        do_r3["aaa"] = False
    if np.array_equal(r3_excitations["aab"][0, :], np.array([1., 1., 1., 1., 1.])):
        do_r3["aab"] = False
    if np.array_equal(r3_excitations["abb"][0, :], np.array([1., 1., 1., 1., 1.])):
        do_r3["abb"] = False

    # Get intermediates
    X = get_cvs_ipeom3_p_intermediates(H, R, r3_excitations, core)
    # update R1
    dR = build_HR_1A(dR, R, Ra, Raa, Rab, r3_excitations, T, H, core)
    # update R2
    dR = build_HR_2A(dR, R, Ra, Raa, Rab, r3_excitations, T, H, X, core)
    dR = build_HR_2B(dR, R, Ra, Raa, Rab, r3_excitations, T, H, X, core)
    # update R3
    if do_r3["aaa"]:
        dR, R, r3_excitations = build_HR_3A(dR, R, r3_excitations, T, X, H)
    if do_r3["aab"]:
        dR, R, r3_excitations = build_HR_3B(dR, R, r3_excitations, T, X, H)
    if do_r3["abb"]:
        dR, R, r3_excitations = build_HR_3C(dR, R, r3_excitations, T, X, H)

    '''for i in range(np.shape(dR.a)[0]):
        if i == core-1:
             continue
        else:
             dR.a[i] = 0
             dR.ab[i,:,:]=0.0
             #dR.abb[i,:,:,:,:]=0.0
    for i in range(np.shape(dR.aa)[0]):
        for j in range(np.shape(dR.aa)[2]):
            if i == core or j == core:
                continue
            else:
                dR.aa[i, :, j] = 0
                #dR.aab[i, :, :, j, :] = 0
    for i in range(np.shape(dR.aaa)[0]):
        for j in range(np.shape(dR.aaa)[0]):
            for k in range(np.shape(dR.aaa)[0]):
                if i == core or j == core or k == core:
                    continue
                else:
                    dR.aaa[i, :, :, j, k] = 0.0

     '''
    
    return dR.flatten()

def build_HR_1A(dR, R, Ra, Raa, Rab, r3_excitations, T, H, core):
    """Calculate the projection <i|[ (H_N e^(T1+T2))_C*(R1h+R2h1p+R3h2p) ]_C|0>."""
    #dR.a = -np.einsum("mi,m->i", H.a.oo, R.a, optimize=True)
    #dR.a -= 0.5 * np.einsum("mnif,mfn->i", H.aa.ooov, R.aa, optimize=True)
    #dR.a -= np.einsum("mnif,mfn->i", H.ab.ooov, R.ab, optimize=True)
    #dR.a += np.einsum("me,iem->i", H.a.ov, R.aa, optimize=True)
    #dR.a += np.einsum("me,iem->i", H.b.ov, R.ab, optimize=True)

    #dR.a = np.zeros_like(R.a)

    dR.a[:core] = - np.einsum("mi,m->i", H.a.oo[:core, :core], Ra["c"], optimize=True)
    dR.a[:core] -= np.einsum("mnif,mfn->i", H.aa.ooov[:core, core:, :core, :], Raa["cv"], optimize=True)
    dR.a[:core] -= 0.5 * np.einsum("mnif,mfn->i", H.aa.ooov[:core, :core, :core, :], Raa["cc"], optimize=True)
    dR.a[:core] -= np.einsum("mnif,mfn->i", H.ab.ooov[:core, :, :core, :], Rab["c"], optimize=True)
    dR.a[:core] += np.einsum("me,iem->i", H.a.ov, Raa["c"], optimize=True)
    dR.a[:core] += np.einsum("me,iem->i", H.b.ov, Rab["c"], optimize=True)

    dR.a = cvs_ipeom3_p_loops.build_hr_1a(
            dR.a,
            R.aaa, r3_excitations["aaa"],
            R.aab, r3_excitations["aab"],
            R.abb, r3_excitations["abb"],
            H.aa.oovv, H.ab.oovv, H.bb.oovv,
    )
    return dR

def build_HR_2A(dR, R, Ra, Raa, Rab, r3_excitations, T, H, X, core):
    """Calculate the projection <ijb|[ (H_N e^(T1+T2))_C*(R1h+R2h1p) ]_C|0>."""
    dR.aa = -0.5 * np.einsum("bmji,m->ibj", H.aa.vooo, R.a, optimize=True)
    dR.aa += 0.5 * np.einsum("be,iej->ibj", H.a.vv, R.aa, optimize=True)
    dR.aa += 0.25 * np.einsum("mnij,mbn->ibj", H.aa.oooo, R.aa, optimize=True)
    I1 = (
        -0.5 * np.einsum("mnef,mfn->e", H.aa.oovv, R.aa, optimize=True)
        - np.einsum("mnef,mfn->e", H.ab.oovv, R.ab, optimize=True)
    )
    dR.aa += 0.5 * np.einsum("e,ebij->ibj", I1, T.aa, optimize=True)
    dR.aa -= np.einsum("mi,mbj->ibj", H.a.oo, R.aa, optimize=True)
    dR.aa += np.einsum("bmje,iem->ibj", H.aa.voov, R.aa, optimize=True)
    dR.aa += np.einsum("bmje,iem->ibj", H.ab.voov, R.ab, optimize=True)
  
    '''dR.aa[:core, :, :core] = - 0.5 * np.einsum("bmji,m->ibj", H.aa.vooo[:, :core, :core, :core], Ra["c"], optimize=True)
    dR.aa[:core, :, core:] -= np.einsum("bmji,m->ibj", H.aa.vooo[:, :core, core:, :core], Ra["c"], optimize=True)
    dR.aa[:core, :, :core] += 0.5 * np.einsum("be,iej->ibj", H.a.vv, Raa["cc"], optimize=True)
    dR.aa[:core, :, core:] += np.einsum("be,iej->ibj", H.a.vv, Raa["cv"], optimize=True)
    dR.aa[:core, :, :core] += np.einsum("mi,jbm->ibj", H.a.oo[:, :core], Raa["c"],  optimize=True) # change (1)
    dR.aa[core:, :, :core] -= np.einsum("mi,mbj->ibj", H.a.oo[:core, core:], Raa["cc"], optimize=True) # (1)
    dR.aa[:core, :, core:] -= np.einsum("mi,mbj->ibj", H.a.oo[:core, :core], Raa["cv"], optimize=True) # (1)
    dR.aa[:core, :, :core] += np.einsum("bmje,iem->ibj", H.aa.voov[:, :, :core, :], Raa["c"], optimize=True) # (2)
    dR.aa[core:, :, :core] -= np.einsum("bmje,mei->ibj", H.aa.voov[:, :core, :core, :], Raa["cv"], optimize=True) # change (2)
    dR.aa[:core, :, core:] += np.einsum("bmje,iem->ibj", H.aa.voov[:, :core, core:, :], Raa["cc"], optimize=True) # (2)
    dR.aa[:core, :, :core] += np.einsum("bmje,iem->ibj", H.ab.voov[:, :, :core, :], Rab["c"], optimize=True) # (3)
    dR.aa[:core, :, core:] += 0.5  * np.einsum("mnij,mbn->ibj", H.aa.oooo[:core, :core, :core, core:], Raa["cc"], optimize=True)
    dR.aa[:core, :, core:] += np.einsum("mnij,mbn->ibj", H.aa.oooo[:core, core:, :core, core:], Raa["cv"], optimize=True)
    dR.aa[:core, :, :core] += 0.25 * np.einsum("mnij,mbn->ibj", H.aa.oooo[:core, :core, :core, :core], Raa["cc"], optimize=True)
    dR.aa[:core, :, :core] += 0.5 * np.einsum("mnij,mbn->ibj", H.aa.oooo[:core, core:, :core, :core], Raa["cv"], optimize=True)
    dR.aa[:core, :, :core] += 0.5 * np.einsum("e,ebij->ibj", X["a"]["v"], T.aa[:, :, :core, :core], optimize=True)
    dR.aa[:core, :, core:] += np.einsum("e,ebij->ibj", X["a"]["v"], T.aa[:, :, :core, core:], optimize=True)
    '''
    dR.aa = cvs_ipeom3_p_loops.build_hr_2a(
            dR.aa,
            R.aaa, r3_excitations["aaa"],
            R.aab, r3_excitations["aab"],
            H.a.ov, H.b.ov,
            H.aa.ooov, H.aa.vovv, H.ab.ooov, H.ab.vovv
    )
    #tmp = - np.einsum("mj,ibm->ibj", H.a.oo[core:, core:], Raa["cv"], optimize=True) # change (1)
    #tmp += np.einsum("bmje,iem->ibj", H.aa.voov[:, core:, core:, :], Raa["cv"], optimize=True) # (2)
    #tmp += np.einsum("bmje,iem->ibj", H.ab.voov[:, :, core:, :], Rab["c"], optimize=True) # (3)

    #dR.aa[:core, :, core:] += tmp
    #dR.aa[core:, :, :core] -= tmp.transpose(2, 1, 0)

    return dR

def build_HR_2B(dR, R, Ra, Raa, Rab, r3_excitations, T, H, X, core):
    """Calculate the projection <ij~b~|[ (H_N e^(T1+T2))_C*(R1h+R2h1p) ]_C|0>."""
    dR.ab = -1.0 * np.einsum("mbij,m->ibj", H.ab.ovoo, R.a, optimize=True)
    dR.ab -= np.einsum("mi,mbj->ibj", H.a.oo, R.ab, optimize=True)
    dR.ab -= np.einsum("mj,ibm->ibj", H.b.oo, R.ab, optimize=True)
    dR.ab += np.einsum("be,iej->ibj", H.b.vv, R.ab, optimize=True)
    dR.ab += np.einsum("mnij,mbn->ibj", H.ab.oooo, R.ab, optimize=True)
    dR.ab += np.einsum("mbej,iem->ibj", H.ab.ovvo, R.aa, optimize=True)
    dR.ab += np.einsum("bmje,iem->ibj", H.bb.voov, R.ab, optimize=True)
    dR.ab -= np.einsum("mbie,mej->ibj", H.ab.ovov, R.ab, optimize=True)
    I1 = (
        -0.5 * np.einsum("mnef,mfn->e", H.aa.oovv, R.aa, optimize=True)
        - np.einsum("mnef,mfn->e", H.ab.oovv, R.ab, optimize=True)
    )
    dR.ab += np.einsum("e,ebij->ibj", I1, T.ab, optimize=True)

    '''dR.ab[:core, :, :] = - np.einsum("mbij,m->ibj", H.ab.ovoo[:core, :, :core, :], Ra["c"], optimize=True)
    dR.ab[:core, :, :] -= np.einsum("mi,mbj->ibj", H.a.oo[:core, :core], Rab["c"], optimize=True)
    dR.ab[:core, :, :] -= np.einsum("mj,ibm->ibj", H.b.oo, Rab["c"], optimize=True)
    dR.ab[:core, :, :] += np.einsum("be,iej->ibj", H.b.vv, Rab["c"], optimize=True)
    dR.ab[:core, :, :] += np.einsum("mnij,mbn->ibj", H.ab.oooo[:core, :, :core, :], Rab["c"], optimize=True)
    dR.ab[:core, :, :] += np.einsum("mbej,iem->ibj", H.ab.ovvo[:core, :, :, :], Raa["cc"], optimize=True)
    dR.ab[:core, :, :] += np.einsum("mbej,iem->ibj", H.ab.ovvo[core:, :, :, :], Raa["cv"], optimize=True)
    dR.ab[:core, :, :] += np.einsum("bmje,iem->ibj", H.bb.voov, Rab["c"], optimize=True)
    dR.ab[:core, :, :] -= np.einsum("mbie,mej->ibj", H.ab.ovov[:core, :, :core, :], Rab["c"], optimize=True)
    dR.ab[:core, :, :] += np.einsum("e,ebij->ibj", X["a"]["v"], T.ab[:, :, :core, :], optimize=True)
    '''
    dR.ab = cvs_ipeom3_p_loops.build_hr_2b(
            dR.ab,
            R.aab, r3_excitations["aab"],
            R.abb, r3_excitations["abb"],
            H.a.ov, H.b.ov,
            H.aa.ooov,
            H.ab.ooov, H.ab.oovo, H.ab.ovvv,
            H.bb.ooov, H.bb.vovv,
    )
    return dR

def build_HR_3A(dR, R, r3_excitations, T, X, H):
    """Calculate the projection <ijkbc|[ (H_N e^(T1+T2))_C*(R1h+R2h1p+R3h2p) ]_C|0>."""
    dR.aaa, R.aaa, r3_excitations["aaa"] = cvs_ipeom3_p_loops.build_hr_3a(
            R.aa,
            R.aaa, r3_excitations["aaa"],
            R.aab, r3_excitations["aab"],
            T.aa,
            H.a.oo, H.a.vv,
            H.aa.vvvv.transpose(2, 3, 0, 1), H.aa.oooo, H.aa.voov, H.aa.vooo, H.aa.vvov,
            H.ab.voov,
            X["aa"]["ovv"], X["aa"]["ooo"],
    )
    return dR, R, r3_excitations

def build_HR_3B(dR, R, r3_excitations, T, X, H):
    """Calculate the projection <ijk~bc~|[ (H_N e^(T1+T2))_C*(R1h+R2h1p+R3h2p) ]_C|0>."""
    dR.aab, R.aab, r3_excitations["aab"] = cvs_ipeom3_p_loops.build_hr_3b(
            R.aa, R.ab,
            R.aaa, r3_excitations["aaa"],
            R.aab, r3_excitations["aab"],
            R.abb, r3_excitations["abb"],
            T.aa, T.ab,
            H.a.oo, H.a.vv, H.b.oo, H.b.vv,
            H.aa.oooo, H.aa.voov, H.aa.vooo,
            H.ab.vvvv.transpose(2, 3, 0, 1), H.ab.oooo, H.ab.voov, H.ab.vovo, H.ab.ovov, H.ab.ovvo,
            H.ab.vooo, H.ab.ovoo, H.ab.vvov, H.ab.vvvo,
            H.bb.voov,
            X["aa"]["ovv"], X["aa"]["ooo"],
            X["ab"]["ovv"], X["ab"]["vvo"], X["ab"]["ooo"],
    )
    return dR, R, r3_excitations

def build_HR_3C(dR, R, r3_excitations, T, X, H):
    """Calculate the projection <ij~k~b~c~|[ (H_N e^(T1+T2))_C*(R1h+R2h1p+R3h2p) ]_C|0>."""
    dR.abb, R.abb, r3_excitations["abb"] = cvs_ipeom3_p_loops.build_hr_3c(
            R.ab,
            R.aab, r3_excitations["aab"],
            R.abb, r3_excitations["abb"],
            T.ab, T.bb,
            H.a.oo, H.b.oo, H.b.vv,
            H.ab.oooo, H.ab.ovov, H.ab.ovvo, H.ab.ovoo,
            H.bb.vvvv.transpose(2, 3, 0, 1), H.bb.oooo, H.bb.voov, H.bb.vooo, H.bb.vvov,
            X["ab"]["ovv"], X["ab"]["vvo"], X["ab"]["ooo"],
    )
    return dR, R, r3_excitations
