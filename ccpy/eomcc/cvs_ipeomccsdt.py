"""Ionization Potential Equation-of-Motion Coupled-Cluster
Method with 1h, 2h-1p, and 3h-2p Excitations on top of CCSD [IP-EOMCCSD(3h-2p)]"""
import numpy as np
from ccpy.eomcc.cvs_ipeom3_intermediates import get_cvs_ipeomccsdt_intermediates, add_v_term
from ccpy.lib.core import cc_loops2

# R.a -> (noa) -> (i)
# R.aa -> (noa,nua,noa) -> (ibj)
# R.ab -> (noa,nub,nob) -> (ib~j~)
# R.aaa -> (noa,nua,nua,noa,noa) -> (ibcjk)
# R.aab -> (noa,nua,nub,noa,nob) -> (ibc~jk~)
# R.abb -> (noa,nub,nub,nob,nob) -> (ib~c~j~k~)

def update(R, omega, H, RHF_symmetry, system):

    R.a, R.aa, R.ab, R.aaa, R.aab, R.abb = cc_loops2.update_r_3h2p(
        R.a,
        R.aa,
        R.ab,
        R.aaa,
        R.aab,
        R.abb,
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
    # Sparsing of R vector
    Ra   = {"c" : {}}
    Raa  = {"c" : {}, "cc" : {}, "cv" : {}}
    Rab  = {"c" : {}}
    Raaa = {"c" : {}, "cc" : {}, "cv" : {}, "ccc" : {}, "ccv" : {}, "cvv" : {}}
    Raab = {"c" : {}, "cc" : {}, "cv" : {}}
    Rabb = {"c" : {}}
    
    Ra["c"]      = R.a[:core]

    Raa["c"]     = R.aa[:core, :, :]
    Raa["cc"]    = R.aa[:core, :, :core]
    Raa["cv"]    = R.aa[:core, :, core:]

    Rab["c"]     = R.ab[:core, :, :]

    Raaa["c"]    = R.aaa[:core, :, :, :, :]
    Raaa["cc"]   = R.aaa[:core, :, :, :core, :]
    Raaa["cv"]   = R.aaa[:core, :, :, core:, :]
    Raaa["ccc"]  = R.aaa[:core, :, :, :core, :core]
    Raaa["ccv"]  = R.aaa[:core, :, :, :core, core:]
    Raaa["cvv"]  = R.aaa[:core, :, :, core:, core:]

    Raab["c"]    = R.aab[:core, :, :, :, :]
    Raab["cc"]   = R.aab[:core, :, :, :core, :]
    Raab["cv"]   = R.aab[:core, :, :, core:, :]

    Rabb["c"]    = R.abb[:core, :, :, :, :]

    # Get intermediates
    X = get_cvs_ipeomccsdt_intermediates(H, R, core)
    # update R1
    dR.a = build_HR_1A(R, Ra, Raa, Rab, Raaa, Raab, Rabb, T, H, core)
    # update R2
    dR.aa = build_HR_2A(R, Ra, Raa, Rab, Raaa, Raab, T, H, X, core)
    dR.ab = build_HR_2B(R, Ra, Raa, Rab, Raab, Rabb, T, H, X, core)
    # update R3
    X = add_v_term(X, H, R, core)
    dR.aaa = build_HR_3A(R, Raa, Raaa, Raab, T, X, H, core)
    dR.aab = build_HR_3B(R, Raa, Rab, Raaa, Raab, Rabb, T, X, H, core)
    dR.abb = build_HR_3C(R, Rab, Raab, Rabb, T, X, H, core)

    return dR.flatten()

def build_HR_1A(R, Ra, Raa, Rab, Raaa, Raab, Rabb, T, H, core):
    """Calculate the projection <I|[ (H_N e^(T1+T2))_C*(R1h+R2h1p+R3h2p) ]_C|0>."""
    X1A = np.zeros_like(R.a)

    X1A[:core] -= np.einsum("mi,m->i", H.a.oo[:core, :core], Ra["c"], optimize=True)
    X1A[:core] -= np.einsum("mnif,mfn->i", H.aa.ooov[:core, core:, :core, :], Raa["cv"], optimize=True)
    X1A[:core] -= 0.5 * np.einsum("mnif,mfn->i", H.aa.ooov[:core, :core, :core, :], Raa["cc"], optimize=True)
    X1A[:core] -= np.einsum("mnif,mfn->i", H.ab.ooov[:core, :, :core, :], Rab["c"], optimize=True)
    X1A[:core] += np.einsum("me,iem->i", H.a.ov, Raa["c"], optimize=True)
    X1A[:core] += np.einsum("me,iem->i", H.b.ov, Rab["c"], optimize=True)
    # additional terms with R3
    X1A[:core] += 0.25 * np.einsum("mnef,iefmn->i", H.aa.oovv, Raaa["c"], optimize=True)
    X1A[:core] += np.einsum("mnef,iefmn->i", H.ab.oovv, Raab["c"], optimize=True)
    X1A[:core] += 0.25 * np.einsum("mnef,iefmn->i", H.bb.oovv, Rabb["c"], optimize=True)

    return X1A

def build_HR_2A(R, Ra, Raa, Rab, Raaa, Raab, T, H, X, core):
    """Calculate the projection <ijb|[ (H_N e^(T1+T2))_C*(R1h+R2h1p) ]_C|0>, where i or j belogs to core index."""
    X2A = np.zeros_like(R.aa)

    X2A[:core, :, :core] -= 0.5 * np.einsum("bmji,m->ibj", H.aa.vooo[:, :core, :core, :core], Ra["c"], optimize=True)
    X2A[:core, :, core:] -= np.einsum("bmji,m->ibj", H.aa.vooo[:, :core, core:, :core], Ra["c"], optimize=True)
    X2A[:core, :, :core] += 0.5 * np.einsum("be,iej->ibj", H.a.vv, Raa["cc"], optimize=True)
    X2A[:core, :, core:] += np.einsum("be,iej->ibj", H.a.vv, Raa["cv"], optimize=True)
    X2A[:core, :, :core] += np.einsum("mi,jbm->ibj", H.a.oo[:, :core], Raa["c"],  optimize=True) # change (1)
    X2A[core:, :, :core] -= np.einsum("mi,mbj->ibj", H.a.oo[:core, core:], Raa["cc"], optimize=True) # (1)
    X2A[:core, :, core:] -= np.einsum("mi,mbj->ibj", H.a.oo[:core, :core], Raa["cv"], optimize=True) # (1)
    X2A[:core, :, :core] += np.einsum("bmje,iem->ibj", H.aa.voov[:, :, :core, :], Raa["c"], optimize=True) # (2)
    X2A[core:, :, :core] -= np.einsum("bmje,mei->ibj", H.aa.voov[:, :core, :core, :], Raa["cv"], optimize=True) # change (2)
    X2A[:core, :, core:] += np.einsum("bmje,iem->ibj", H.aa.voov[:, :core, core:, :], Raa["cc"], optimize=True) # (2)
    X2A[:core, :, :core] += np.einsum("bmje,iem->ibj", H.ab.voov[:, :, :core, :], Rab["c"], optimize=True) # (3)
    X2A[:core, :, core:] += 0.5  * np.einsum("mnij,mbn->ibj", H.aa.oooo[:core, :core, :core, core:], Raa["cc"], optimize=True)
    X2A[:core, :, core:] += np.einsum("mnij,mbn->ibj", H.aa.oooo[:core, core:, :core, core:], Raa["cv"], optimize=True)
    X2A[:core, :, :core] += 0.25 * np.einsum("mnij,mbn->ibj", H.aa.oooo[:core, :core, :core, :core], Raa["cc"], optimize=True)
    X2A[:core, :, :core] += 0.5 * np.einsum("mnij,mbn->ibj", H.aa.oooo[:core, core:, :core, :core], Raa["cv"], optimize=True)
    X2A[:core, :, :core] += 0.5 * np.einsum("e,ebij->ibj", X["a"]["v"], T.aa[:, :, :core, :core], optimize=True)
    X2A[:core, :, core:] += np.einsum("e,ebij->ibj", X["a"]["v"], T.aa[:, :, :core, core:], optimize=True)
    # additional terms with R3
    X2A[:core, :, :core] -= 0.5 * np.einsum("mnjf,ibfmn->ibj", H.aa.ooov[:, :, :core, :], Raaa["c"], optimize=True) # (4)
    X2A[:core, :, core:] -= 0.5 * np.einsum("mnjf,ibfmn->ibj", H.aa.ooov[:core, :core, core:, :], Raaa["ccc"], optimize=True) # (4)
    X2A[:core, :, core:] -= np.einsum("mnjf,ibfmn->ibj", H.aa.ooov[:core, core:, core:, :], Raaa["ccv"], optimize=True) # (4)
    X2A[core:, :, :core] += 0.5 * np.einsum("mnjf,nbfmi->ibj", H.aa.ooov[:core, :core, :core, :], Raaa["ccv"], optimize=True) # change (4)
    X2A[core:, :, :core] += np.einsum("mnjf,mbfin->ibj", H.aa.ooov[:core, core:, :core, :], Raaa["cvv"], optimize=True) # change (4)
    X2A[:core, :, :core] -= np.einsum("mnjf,ibfmn->ibj", H.ab.ooov[:, :, :core, :], Raab["c"], optimize=True) # (5)
    X2A[:core, :, core:] -= np.einsum("mnjf,ibfmn->ibj", H.ab.ooov[:core, :, core:, :], Raab["cc"], optimize=True) # (5)
    X2A[core:, :, :core] += np.einsum("mnjf,mbfin->ibj", H.ab.ooov[:core, :, :core, :], Raab["cv"], optimize=True) # (5)
    X2A[:core, :, :core] += 0.5 * np.einsum("me,ibejm->ibj", H.a.ov, Raaa["cc"], optimize=True)
    X2A[:core, :, core:] += np.einsum("me,ibejm->ibj", H.a.ov, Raaa["cv"], optimize=True)
    X2A[:core, :, :core] += 0.5 * np.einsum("me,ibejm->ibj", H.b.ov, Raab["cc"], optimize=True)
    X2A[:core, :, core:] += np.einsum("me,ibejm->ibj", H.b.ov, Raab["cv"], optimize=True)
    X2A[:core, :, :core] += 0.25 * np.einsum("bnef,iefjn->ibj", H.aa.vovv, Raaa["cc"], optimize=True)
    X2A[:core, :, core:] += 0.5 * np.einsum("bnef,iefjn->ibj", H.aa.vovv, Raaa["cv"], optimize=True)
    X2A[:core, :, :core] += 0.5 * np.einsum("bnef,iefjn->ibj", H.ab.vovv, Raab["cc"], optimize=True)
    X2A[:core, :, core:] += np.einsum("bnef,iefjn->ibj", H.ab.vovv, Raab["cv"], optimize=True)

    X2A -= np.transpose(X2A, (2, 1, 0))

    tmp = - np.einsum("mj,ibm->ibj", H.a.oo[core:, core:], Raa["cv"], optimize=True) # change (1)
    tmp += np.einsum("bmje,iem->ibj", H.aa.voov[:, core:, core:, :], Raa["cv"], optimize=True) # (2)
    tmp += np.einsum("bmje,iem->ibj", H.ab.voov[:, :, core:, :], Rab["c"], optimize=True) # (3)
    # additional terms with R3
    tmp -= 0.5 * np.einsum("mnjf,ibfmn->ibj", H.aa.ooov[core:, core:, core:, :], Raaa["cvv"], optimize=True) # (4)
    tmp -= np.einsum("mnjf,ibfmn->ibj", H.ab.ooov[core:, :, core:, :], Raab["cv"], optimize=True) # (5)

    X2A[:core, :, core:] += tmp
    X2A[core:, :, :core] -= tmp.transpose(2, 1, 0)

    # terms with T3 -> Don't include; they are part of h(vvov)*R1 and h(vooo)*R1, which are taken care of
    # X2A += 0.25 * np.einsum("fne,ebfijn->ibj", X["aa"]["vov"], T.aaa, optimize=True)
    # X2A += 0.5 * np.einsum("fne,ebfijn->ibj", X["ab"]["vov"], T.aab, optimize=True)

    return X2A

def build_HR_2B(R, Ra, Raa, Rab, Raab, Rabb, T, H, X, core):
    """Calculate the projection <Ij~b~|[ (H_N e^(T1+T2))_C*(R1h+R2h1p) ]_C|0>."""
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
    X2B[:core, :, :] += np.einsum("e,ebij->ibj", X["a"]["v"], T.ab[:, :, :core, :], optimize=True)
    # additional terms with R3
    X2B[:core, :, :] += np.einsum("me,iebmj->ibj", H.a.ov, Raab["c"], optimize=True)
    X2B[:core, :, :] += np.einsum("me,ibejm->ibj", H.b.ov, Rabb["c"], optimize=True)
    X2B[:core, :, :] += np.einsum("nbfe,ifenj->ibj", H.ab.ovvv, Raab["c"], optimize=True)
    X2B[:core, :, :] += 0.5 * np.einsum("bnef,iefjn->ibj", H.bb.vovv, Rabb["c"], optimize=True)
    X2B[:core, :, :] -= 0.5 * np.einsum("mnif,mfbnj->ibj", H.aa.ooov[:core, :core, :core, :], Raab["cc"], optimize=True)
    X2B[:core, :, :] -= np.einsum("mnif,mfbnj->ibj", H.aa.ooov[:core, core:, :core, :], Raab["cv"], optimize=True)
    X2B[:core, :, :] -= np.einsum("mnif,mfbnj->ibj", H.ab.ooov[:core, :, :core, :], Rabb["c"], optimize=True)
    X2B[:core, :, :] -= np.einsum("nmfj,ifbnm->ibj", H.ab.oovo, Raab["c"], optimize=True)
    X2B[:core, :, :] -= 0.5 * np.einsum("mnjf,ifbnm->ibj", H.bb.ooov, Rabb["c"], optimize=True)

    # terms with T3 -> Don't include; they are part of h(vvov)*R1 and h(vooo)*R1, which are taken care of
    # X2B += 0.5 * np.einsum("fne,efbinj->ibj", X["aa"]["vov"], T.aab, optimize=True)
    # X2B += np.einsum("fne,ebfijn->ibj", X["ab"]["vov"], T.abb, optimize=True)

    return X2B

def build_HR_3A(R, Raa, Raaa, Raab, T, X, H, core):
    """Calculate the projection <ijkbc|[ (H_N e^(T1+T2))_C*(R1h+R2h1p+R3h2p) ]_C|0>, where i or j or k belongs to core index."""
    X3A   = np.zeros_like(R.aaa)
    tmp_cvv = np.zeros_like(R.aaa)
    tmp_vcc = np.zeros_like(R.aaa)
    tmp_vvv = np.zeros_like(R.aaa)

    tmp_cvv[:core, :, :, core:, core:] =(
                                        - 0.5 * np.einsum("mj,ibcmk->ibcjk", H.a.oo[core:, core:], Raaa["cvv"], optimize=True) # (1)
                                        + np.einsum("bmje,iecmk->ibcjk", H.aa.voov[:, core:, core:, :], Raaa["cvv"], optimize=True) # (3)
                                        + np.einsum("bmje,icekm->ibcjk", H.ab.voov[:, :, core:, :], Raab["cv"], optimize=True) # (4)
                                        # moment-like terms
                                        + 0.5 * np.einsum("cbke,iej->ibcjk", H.aa.vvov[:, :, core:, :], Raa["cv"], optimize=True) # (6)
                                        # 3-body Hbar terms factorized using intermediates
                                        - 0.5 * np.einsum("imj,bcmk->ibcjk", X["aa"]["oooIj/k"], T.aa[:, :, :, core:],  optimize=True) # (10)
                                        - 0.5 * np.einsum("imj,bcmk->ibcjk", X["aa"]["oooi_j/k"][:core, :, core:], T.aa[:, :, :, core:], optimize=True) # (7)
                                        )
    tmp_vcc[core:, :, :, :core, :core] =(
                                        - 0.25 * np.einsum("mnik,jbcmn->ibcjk", H.aa.oooo[core:, core:, core:, :core], Raaa["cvv"], optimize=True) # (2)
                                        # moment-like terms
                                        + np.einsum("cmij,kbm->ibcjk", H.aa.vooo[:, core:, core:, :core], Raa["cv"], optimize=True) # (5)
                                        # 3-body Hbar terms factorized using intermediates
                                        - 0.5 * np.einsum("jmk,bcmi->ibcjk", X["aa"]["oooi_j/k"][:core, :, :core], T.aa[:, :, :, core:], optimize=True) # (7)
                                        + 0.5 * np.einsum("jmi,bcmk->ibcjk", X["aa"]["oooi_j/k"][:core, :, core:], T.aa[:, :, :, :core], optimize=True) # (7)
                                        - np.einsum("jbe,ecik->ibcjk", X["aa"]["ovvI/jk"], T.aa[:, :, core:, :core], optimize=True) # (11)
                                        # parts with T3
                                        - 0.5 * np.einsum("jem,ebcmik->ibcjk", X["aa"]["ovoI/jk"], T.aaa[:, :, :, :, core:, :core], optimize=True) # [1]
                                        - 0.5 * np.einsum("jem,bceikm->ibcjk", X["ab"]["ovoI/jk"], T.aab[:, :, :, core:, :core, :], optimize=True) # [2]
                                        )
    tmp_vvv[:core, :, :, core:, core:] =(
                                        + 0.25 * np.einsum("mnjk,ibcmn->ibcjk", H.aa.oooo[core:, core:, core:, core:], Raaa["cvv"], optimize=True) # (2)
                                        # moment-like terms
                                        - np.einsum("cmkj,ibm->ibcjk", H.aa.vooo[:, core:, core:, core:], Raa["cv"], optimize=True) # (5)
                                        # 3-body Hbar terms factorized using intermediates
                                        + np.einsum("ibe,ecjk->ibcjk", X["aa"]["ovvI/jk"], T.aa[:, :, core:, core:], optimize=True) # (12)
                                        # parts with T3
                                        + 0.5 * np.einsum("iem,ebcmjk->ibcjk", X["aa"]["ovoI/jk"], T.aaa[:, :, :, :, core:, core:], optimize=True) # [1]
                                        + 0.5 * np.einsum("iem,bcejkm->ibcjk", X["ab"]["ovoI/jk"], T.aab[:, :, :, core:, core:, :], optimize=True) # [2]
                                        )
    X3A[:core, :, :, :core, :core] += 0.25 * np.einsum("mj,ibckm->ibcjk", H.a.oo[:, :core], Raaa["cc"], optimize=True) # change (1)
    X3A[:core, :, :, :core, core:] += 0.5 * np.einsum("mj,ibckm->ibcjk", H.a.oo[:, :core], Raaa["cv"], optimize=True) # change (1)
    X3A[:core, :, :, core:, :core] += 0.25 * np.einsum("mj,ibckm->ibcjk", H.a.oo[:, core:], Raaa["cc"], optimize=True) # change (1)
    X3A[:core, :, :, core:, core:] -= 0.5 * np.einsum("mj,ibcmk->ibcjk", H.a.oo[:core, core:], Raaa["ccv"], optimize=True) # (1)
    X3A[core:, :, :, :core, core:] += 0.25 * np.einsum("mj,mbcik->ibcjk", H.a.oo[:core, :core], Raaa["cvv"], optimize=True) # change (1)
    X3A[:core, :, :, :core, :core] += (2.0/12.0) * np.einsum("be,iecjk->ibcjk", H.a.vv, Raaa["ccc"], optimize=True)
    X3A[core:, :, :, :core, :core] -= 0.5 * np.einsum("be,kecji->ibcjk", H.a.vv, Raaa["ccv"], optimize=True) # change
    X3A[:core, :, :, core:, core:] += 0.5 * np.einsum("be,iecjk->ibcjk", H.a.vv, Raaa["cvv"], optimize=True)
    X3A[:core, :, :, :core, :core] += (3.0/24.0) * np.einsum("mnjk,ibcmn->ibcjk", H.aa.oooo[:, :, :core, :core], Raaa["c"], optimize=True) # (2)
    X3A[core:, :, :, :core, :core] -= (3.0/24.0) * np.einsum("mnjk,nbcmi->ibcjk", H.aa.oooo[:core, :core, :core, :core], Raaa["ccv"], optimize=True) # change (2)
    X3A[core:, :, :, :core, :core] -= 0.25 * np.einsum("mnjk,mbcin->ibcjk", H.aa.oooo[:core, core:, :core, :core], Raaa["cvv"], optimize=True) # change (2)
    X3A[:core, :, :, core:, core:] += (3.0/24.0) * np.einsum("mnjk,ibcmn->ibcjk", H.aa.oooo[:core, :core, core:, core:], Raaa["ccc"], optimize=True) # (2)
    X3A[:core, :, :, core:, core:] += 0.25 * np.einsum("mnjk,ibcmn->ibcjk", H.aa.oooo[:core, core:, core:, core:], Raaa["ccv"], optimize=True) # (2)
    X3A[:core, :, :, :core, core:] += 0.25 * np.einsum("mnjk,ibcmn->ibcjk", H.aa.oooo[:core, :core, :core, core:], Raaa["ccc"], optimize=True) # (2)
    X3A[:core, :, :, :core, core:] += 0.5 * np.einsum("mnjk,ibcmn->ibcjk", H.aa.oooo[:core, core:, :core, core:], Raaa["ccv"], optimize=True) # (2)
    X3A[core:, :, :, :core, core:] -= 0.25 * np.einsum("mnjk,nbcmi->ibcjk", H.aa.oooo[:core, :core, :core, core:], Raaa["ccv"], optimize=True) # change (2)
    X3A[core:, :, :, :core, core:] -= 0.5 * np.einsum("mnjk,mbcin->ibcjk", H.aa.oooo[:core, core:, :core, core:], Raaa["cvv"], optimize=True) # change (2)
    X3A[:core, :, :, :core, :core] += (1.0/24.0) * np.einsum("bcef,iefjk->ibcjk", H.aa.vvvv, Raaa["ccc"], optimize=True)
    X3A[core:, :, :, :core, :core] -= (3.0/24.0) * np.einsum("bcef,kefji->ibcjk", H.aa.vvvv, Raaa["ccv"], optimize=True) # change
    X3A[:core, :, :, core:, core:] += (3.0/24.0) * np.einsum("bcef,iefjk->ibcjk", H.aa.vvvv, Raaa["cvv"], optimize=True)
    X3A[:core, :, :, :core, :core] -= 0.5 * np.einsum("bmje,ieckm->ibcjk", H.aa.voov[:, :, :core, :], Raaa["cc"], optimize=True) # change (3)
    X3A[:core, :, :, :core, core:] -= np.einsum("bmje,ieckm->ibcjk", H.aa.voov[:, :, :core, :], Raaa["cv"], optimize=True) # change (3)
    X3A[:core, :, :, core:, :core] -= 0.5 * np.einsum("bmje,ieckm->ibcjk", H.aa.voov[:, :, core:, :], Raaa["cc"], optimize=True) # change (3)
    X3A[:core, :, :, core:, core:] += np.einsum("bmje,iecmk->ibcjk", H.aa.voov[:, :core, core:, :], Raaa["ccv"], optimize=True) # (3)
    X3A[core:, :, :, :core, core:] -= 0.5 * np.einsum("bmje,mecik->ibcjk", H.aa.voov[:, :core, :core, :], Raaa["cvv"], optimize=True) # (3)
    X3A[:core, :, :, :core, :core] += 0.5 * np.einsum("bmje,icekm->ibcjk", H.ab.voov[:, :, :core, :], Raab["cc"], optimize=True) # (4)
    X3A[:core, :, :, :core, core:] += np.einsum("bmje,icekm->ibcjk", H.ab.voov[:, :, :core, :], Raab["cv"], optimize=True) # (4)
    X3A[:core, :, :, core:, :core] += 0.5 * np.einsum("bmje,icekm->ibcjk", H.ab.voov[:, :, core:, :], Raab["cc"], optimize=True) # (4)
    # moment-like terms
    X3A[:core, :, :, :core, :core] -= 0.5 * np.einsum("cmkj,ibm->ibcjk", H.aa.vooo[:, :, :core, :core], Raa["c"], optimize=True) # (5)
    X3A[:core, :, :, :core, core:] -= np.einsum("cmkj,ibm->ibcjk", H.aa.vooo[:, :core, core:, :core], Raa["cc"], optimize=True) # (5)
    X3A[core:, :, :, :core, core:] += np.einsum("cmkj,mbi->ibcjk", H.aa.vooo[:, :core, core:, :core], Raa["cv"], optimize=True) # change (5)
    X3A[core:, :, :, :core, :core] += 0.5 * np.einsum("cmkj,mbi->ibcjk", H.aa.vooo[:, :core, :core, :core], Raa["cv"], optimize=True) # change (5)
    X3A[:core, :, :, core:, core:] -= 0.5 * np.einsum("cmkj,ibm->ibcjk", H.aa.vooo[:, :core, core:, core:], Raa["cc"], optimize=True) # (5)
    X3A[:core, :, :, :core, :core] += 0.25 * np.einsum("cbke,iej->ibcjk", H.aa.vvov[:, :, :core, :], Raa["cc"], optimize=True) # (6)
    X3A[:core, :, :, :core, core:] += 0.25 * np.einsum("cbke,iej->ibcjk", H.aa.vvov[:, :, core:, :], Raa["cc"], optimize=True) # (6)
    X3A[:core, :, :, core:, :core] += 0.5 * np.einsum("cbke,iej->ibcjk", H.aa.vvov[:, :, :core, :], Raa["cv"], optimize=True) # (6)
    # 3-body Hbar terms factorized using intermediates
    X3A[:core, :, :, :core, :core] -= 0.5 * np.einsum("imj,bcmk->ibcjk", X["aa"]["oooi_j/k"][:core, :, :core], T.aa[:, :, :, :core], optimize=True) # (7) 
    X3A[:core, :, :, :core, :core] -= 0.25 * np.einsum("imj,bcmk->ibcjk", X["aa"]["oooIJ/k"], T.aa[:, :, :, :core], optimize=True) # (8)
    X3A[:core, :, :, :core, core:] -= 0.25 * np.einsum("imj,bcmk->ibcjk", X["aa"]["oooIJ/k"], T.aa[:, :, :, core:], optimize=True) #  (8)
    X3A[:core, :, :, core:, :core] -= 0.5 * np.einsum("imj,bcmk->ibcjk", X["aa"]["oooij/k"][:core, :, core:], T.aa[:, :, :, :core], optimize=True) #  (9)
    X3A[:core, :, :, core:, core:] -= 0.5 * np.einsum("imj,bcmk->ibcjk", X["aa"]["oooij/k"][:core, :, core:], T.aa[:, :, :, core:], optimize=True) #  (9)
    X3A[core:, :, :, core:, :core] -= 0.25 * np.einsum("imj,bcmk->ibcjk", X["aa"]["oooij/k"][core:, :, core:], T.aa[:, :, :, :core], optimize=True) #  (9)
    X3A[:core, :, :, core:, :core] -= 0.5 * np.einsum("imj,bcmk->ibcjk", X["aa"]["oooIj/k"], T.aa[:, :, :, :core], optimize=True) # (10)
    X3A[:core, :, :, :core, :core] += 0.5 * np.einsum("ibe,ecjk->ibcjk", X["aa"]["ovvi/jk"][:core, :, :], T.aa[:, :, :core, :core], optimize=True) # (11)
    X3A[:core, :, :, core:, core:] += 0.5 * np.einsum("ibe,ecjk->ibcjk", X["aa"]["ovvi/jk"][:core, :, :], T.aa[:, :, core:, core:], optimize=True) # (11)
    X3A[:core, :, :, :core, core:] += np.einsum("ibe,ecjk->ibcjk", X["aa"]["ovvi/jk"][:core, :, :], T.aa[:, :, :core, core:], optimize=True) # (11)
    X3A[core:, :, :, :core, core:] += np.einsum("ibe,ecjk->ibcjk", X["aa"]["ovvi/jk"][core:, :, :], T.aa[:, :, :core, core:], optimize=True) # (11)
    X3A[core:, :, :, :core, :core] += 0.5 * np.einsum("ibe,ecjk->ibcjk", X["aa"]["ovvi/jk"][core:, :, :], T.aa[:, :, :core, :core], optimize=True) # (11)
    X3A[:core, :, :, :core, :core] += 0.5 * np.einsum("ibe,ecjk->ibcjk", X["aa"]["ovvI/jk"], T.aa[:, :, :core, :core], optimize=True) # (12)
    # parts with T3
    X3A[:core, :, :, :core, :core] += 0.25 * np.einsum("iem,ebcmjk->ibcjk", X["aa"]["ovoi/jk"][:core, :, :], T.aaa[:, :, :, :, :core, :core], optimize=True) # [1]
    X3A[:core, :, :, core:, core:] += 0.25 * np.einsum("iem,ebcmjk->ibcjk", X["aa"]["ovoi/jk"][:core, :, :], T.aaa[:, :, :, :, core:, core:], optimize=True) # [1]
    X3A[core:, :, :, :core, core:] += 0.5 * np.einsum("iem,ebcmjk->ibcjk", X["aa"]["ovoi/jk"][core:, :, :], T.aaa[:, :, :, :, :core, core:], optimize=True) # [1]
    X3A[:core, :, :, :core, core:] += 0.5 * np.einsum("iem,ebcmjk->ibcjk", X["aa"]["ovoi/jk"][:core, :, :], T.aaa[:, :, :, :, :core, core:], optimize=True) # [1]
    X3A[core:, :, :, :core, :core] += 0.25 * np.einsum("iem,ebcmjk->ibcjk", X["aa"]["ovoi/jk"][core:, :, :], T.aaa[:, :, :, :, :core, :core], optimize=True) # [1]
    X3A[:core, :, :, :core, :core] += 0.25 * np.einsum("iem,ebcmjk->ibcjk", X["aa"]["ovoI/jk"], T.aaa[:, :, :, :, :core, :core], optimize=True) # [1]
    X3A[:core, :, :, :core, :core] += 0.25 * np.einsum("iem,bcejkm->ibcjk", X["ab"]["ovoi/jk"][:core, :, :], T.aab[:, :, :, :core, :core, :], optimize=True) # [2]
    X3A[:core, :, :, core:, core:] += 0.25 * np.einsum("iem,bcejkm->ibcjk", X["ab"]["ovoi/jk"][:core, :, :], T.aab[:, :, :, core:, core:, :], optimize=True) # [2]
    X3A[core:, :, :, :core, core:] += 0.5 * np.einsum("iem,bcejkm->ibcjk", X["ab"]["ovoi/jk"][core:, :, :], T.aab[:, :, :, :core, core:, :], optimize=True) # [2]
    X3A[:core, :, :, :core, core:] += 0.5 * np.einsum("iem,bcejkm->ibcjk", X["ab"]["ovoi/jk"][:core, :, :], T.aab[:, :, :, :core, core:, :], optimize=True) # [2]
    X3A[core:, :, :, :core, :core] += 0.25 * np.einsum("iem,bcejkm->ibcjk", X["ab"]["ovoi/jk"][core:, :, :], T.aab[:, :, :, :core, :core, :], optimize=True) # [2]
    X3A[:core, :, :, :core, :core] += 0.25 * np.einsum("iem,bcejkm->ibcjk", X["ab"]["ovoI/jk"], T.aab[:, :, :, :core, :core, :], optimize=True) # [2]
    X3A[:core, :, :, :core, :core] += (2.0 / 24.0) * np.einsum("bef,fecijk->ibcjk", X["aa"]["vvv"], T.aaa[:, :, :, :core, :core, :core], optimize=True) # [3]
    X3A[:core, :, :, core:, core:] += 0.25 * np.einsum("bef,fecijk->ibcjk", X["aa"]["vvv"], T.aaa[:, :, :, :core, core:, core:], optimize=True) # [3]
    X3A[core:, :, :, :core, :core] += 0.25 * np.einsum("bef,fecijk->ibcjk", X["aa"]["vvv"], T.aaa[:, :, :, core:, :core, :core], optimize=True) # [3]
    X3A[:core, :, :, :core, :core] += (1.0 / 12.0) * np.einsum("e,ebcijk->ibcjk", X["a"]["v"], T.aaa[:, :, :, :core, :core, :core], optimize=True) # [4]
    X3A[:core, :, :, core:, core:] += 0.25 * np.einsum("e,ebcijk->ibcjk", X["a"]["v"], T.aaa[:, :, :, :core, core:, core:], optimize=True) # [4]
    X3A[core:, :, :, :core, :core] += 0.25 * np.einsum("e,ebcijk->ibcjk", X["a"]["v"], T.aaa[:, :, :, core:, :core, :core], optimize=True) # [4]

    X3A -= np.transpose(X3A, (3, 1, 2, 0, 4)) + np.transpose(X3A, (4, 1, 2, 3, 0)) - tmp_cvv - tmp_vcc # antisymmetrize A(i/jk) + some other terms
    X3A -= np.transpose(X3A, (0, 1, 2, 4, 3)) # antisymmetrize A(jk)
    
    X3A -= np.transpose(tmp_cvv, (4, 1, 2, 3, 0)) + np.transpose(tmp_vcc, (4, 1, 2, 3, 0))
    X3A += np.transpose(np.transpose(tmp_cvv, (4, 1, 2, 3, 0)), (3, 1, 2, 0, 4)) + np.transpose(np.transpose(tmp_vcc, (4, 1, 2, 3, 0)), (3, 1, 2, 0, 4)) # antisymmetrize A(ij)

    X3A -= np.transpose(tmp_cvv, (3, 1, 2, 0, 4)) + np.transpose(tmp_vcc, (3, 1, 2, 0, 4))
    X3A += np.transpose(np.transpose(tmp_cvv, (3, 1, 2, 0, 4)), (4, 1, 2, 3, 0)) + np.transpose(np.transpose(tmp_vcc, (3, 1, 2, 0, 4)), (4, 1, 2, 3, 0)) # antisymmetrize A(ik)

    X3A += tmp_vvv - np.transpose(tmp_vvv, (3, 1, 2, 0, 4)) - np.transpose(tmp_vvv, (4, 1, 2, 3, 0))

    X3A -= np.transpose(X3A, (0, 2, 1, 3, 4)) # antisymmetrize A(bc)

    return X3A


def build_HR_3B(R, Raa, Rab, Raaa, Raab, Rabb, T, X, H, core):
    """Calculate the projection <ijk~bc~|[ (H_N e^(T1+T2))_C*(R1h+R2h1p+R3h2p) ]_C|0>, where i or j belong to core index."""
    X3B = np.zeros_like(R.aab)

    X3B[:core, :, :, :core, :] -= np.einsum("mj,ibcmk->ibcjk", H.a.oo[:, :core], Raab["c"], optimize=True) # (1)
    X3B[:core, :, :, core:, :] -= np.einsum("mj,ibcmk->ibcjk", H.a.oo[:core, core:], Raab["cc"], optimize=True) # (1)
    X3B[core:, :, :, :core, :] += np.einsum("mj,mbcik->ibcjk", H.a.oo[:core, :core], Raab["cv"], optimize=True) # (1)
    X3B[:core, :, :, :core, :] -= 0.5 * np.einsum("mk,ibcjm->ibcjk", H.b.oo, Raab["cc"], optimize=True) 
    X3B[:core, :, :, core:, :] -= np.einsum("mk,ibcjm->ibcjk", H.b.oo, Raab["cv"], optimize=True) 
    X3B[:core, :, :, :core, :] += 0.5 * np.einsum("be,iecjk->ibcjk", H.a.vv, Raab["cc"], optimize=True) 
    X3B[:core, :, :, core:, :] += np.einsum("be,iecjk->ibcjk", H.a.vv, Raab["cv"], optimize=True) 
    X3B[:core, :, :, :core, :] += 0.5 * np.einsum("ce,ibejk->ibcjk", H.b.vv, Raab["cc"], optimize=True) 
    X3B[:core, :, :, core:, :] += np.einsum("ce,ibejk->ibcjk", H.b.vv, Raab["cv"], optimize=True) 
    X3B[:core, :, :, :core, :] += np.einsum("mnjk,ibcmn->ibcjk", H.ab.oooo[:, :, :core, :], Raab["c"], optimize=True) # (2)
    X3B[:core, :, :, core:, :] += np.einsum("mnjk,ibcmn->ibcjk", H.ab.oooo[:core, :, core:, :], Raab["cc"], optimize=True) # (2)
    X3B[core:, :, :, :core, :] -= np.einsum("mnjk,mbcin->ibcjk", H.ab.oooo[:core, :, :core, :], Raab["cv"], optimize=True) # change (2)
    X3B[:core, :, :, :core, :] += 0.25 * np.einsum("mnij,mbcnk->ibcjk", H.aa.oooo[:core, :core, :core, :core], Raab["cc"], optimize=True) 
    X3B[:core, :, :, :core, :] += 0.5 * np.einsum("mnij,mbcnk->ibcjk", H.aa.oooo[:core, core:, :core, :core], Raab["cv"], optimize=True) 
    X3B[:core, :, :, core:, :] += 0.5 * np.einsum("mnij,mbcnk->ibcjk", H.aa.oooo[:core, :core, :core, core:], Raab["cc"], optimize=True)
    X3B[:core, :, :, core:, :] += np.einsum("mnij,mbcnk->ibcjk", H.aa.oooo[:core, core:, :core, core:], Raab["cv"], optimize=True) 
    X3B[:core, :, :, :core, :] += 0.5 * np.einsum("bcef,iefjk->ibcjk", H.ab.vvvv, Raab["cc"], optimize=True) 
    X3B[:core, :, :, core:, :] += np.einsum("bcef,iefjk->ibcjk", H.ab.vvvv, Raab["cv"], optimize=True)
    X3B[:core, :, :, :core, :] += 0.5 * np.einsum("mcek,ibejm->ibcjk", H.ab.ovvo, Raaa["cc"], optimize=True) 
    X3B[:core, :, :, core:, :] += np.einsum("mcek,ibejm->ibcjk", H.ab.ovvo, Raaa["cv"], optimize=True) 
    X3B[:core, :, :, :core, :] += 0.5 * np.einsum("cmke,ibejm->ibcjk", H.bb.voov, Raab["cc"], optimize=True)
    X3B[:core, :, :, core:, :] += np.einsum("cmke,ibejm->ibcjk", H.bb.voov, Raab["cv"], optimize=True) 
    X3B[:core, :, :, :core, :] += np.einsum("bmje,iecmk->ibcjk", H.aa.voov[:, :, :core, :], Raab["c"], optimize=True) # (3)
    X3B[:core, :, :, core:, :] += np.einsum("bmje,iecmk->ibcjk", H.aa.voov[:, :core, core:, :], Raab["cc"], optimize=True) # (3)
    X3B[core:, :, :, :core, :] -= np.einsum("bmje,mecik->ibcjk", H.aa.voov[:, :core, :core, :], Raab["cv"], optimize=True) # change (3)
    X3B[:core, :, :, :core, :] += np.einsum("bmje,iecmk->ibcjk", H.ab.voov[:, :, :core, :], Rabb["c"], optimize=True) # (4)
    X3B[:core, :, :, :core, :] -= np.einsum("mcje,ibemk->ibcjk", H.ab.ovov[:, :, :core, :], Raab["c"], optimize=True) # (5)
    X3B[:core, :, :, core:, :] -= np.einsum("mcje,ibemk->ibcjk", H.ab.ovov[:core, :, core:, :], Raab["cc"], optimize=True) # (5)
    X3B[core:, :, :, :core, :] += np.einsum("mcje,mbeik->ibcjk", H.ab.ovov[:core, :, :core, :], Raab["cv"], optimize=True) # change (5)
    X3B[:core, :, :, :core, :] -= 0.5 * np.einsum("bmek,iecjm->ibcjk", H.ab.vovo, Raab["cc"], optimize=True) 
    X3B[:core, :, :, core:, :] -= np.einsum("bmek,iecjm->ibcjk", H.ab.vovo, Raab["cv"], optimize=True) 
    # moment-like terms
    X3B[:core, :, :, :core, :] -= np.einsum("mcjk,ibm->ibcjk", H.ab.ovoo[:, :, :core, :], Raa["c"], optimize=True) # (6)
    X3B[:core, :, :, core:, :] -= np.einsum("mcjk,ibm->ibcjk", H.ab.ovoo[:core, :, core:, :], Raa["cc"], optimize=True) # (6)
    X3B[core:, :, :, :core, :] += np.einsum("mcjk,mbi->ibcjk", H.ab.ovoo[:core, :, :core, :], Raa["cv"], optimize=True) # change (6)
    X3B[:core, :, :, :core, :] -= 0.5 * np.einsum("bmji,mck->ibcjk", H.aa.vooo[:, :core, :core, :core], Rab["c"], optimize=True)
    X3B[:core, :, :, core:, :] -= np.einsum("bmji,mck->ibcjk", H.aa.vooo[:, :core, core:, :core], Rab["c"], optimize=True)
    X3B[:core, :, :, :core, :] -= np.einsum("bmjk,icm->ibcjk", H.ab.vooo[:, :, :core, :], Rab["c"], optimize=True) # (7)
    X3B[:core, :, :, :core, :] += np.einsum("bcje,iek->ibcjk", H.ab.vvov[:, :, :core, :], Rab["c"], optimize=True) # (8)
    X3B[:core, :, :, :core, :] += 0.5 * np.einsum("bcek,iej->ibcjk", H.ab.vvvo, Raa["cc"], optimize=True)
    X3B[:core, :, :, core:, :] += np.einsum("bcek,iej->ibcjk", H.ab.vvvo, Raa["cv"], optimize=True)
    # 3-body Hbar terms factorized using intermediates
    X3B[:core, :, :, :core, :] += 0.5 * np.einsum("eck,ebij->ibcjk", X["ab"]["vvo"], T.aa[:, :, :core, :core], optimize=True)
    X3B[:core, :, :, core:, :] += np.einsum("eck,ebij->ibcjk", X["ab"]["vvo"], T.aa[:, :, :core, core:], optimize=True)
    X3B[:core, :, :, :core, :] -= 0.5 * np.einsum("imj,bcmk->ibcjk", X["aa"]["oooIJ/k"], T.ab, optimize=True)
    X3B[:core, :, :, core:, :] -= np.einsum("imj,bcmk->ibcjk", X["aa"]["oooIj/k"], T.ab, optimize=True) # (9)
    X3B[:core, :, :, core:, :] -= np.einsum("imj,bcmk->ibcjk", X["aa"]["oooij/k"][:core, :, core:], T.ab, optimize=True) # (10)
    X3B[:core, :, :, :core, :] -= np.einsum("imj,bcmk->ibcjk", X["aa"]["oooi_j/k"][:core, :, :core], T.ab, optimize=True) # (11)
    X3B[:core, :, :, :core, :] -= np.einsum("imk,bcjm->ibcjk", X["ab"]["oooi/jk"][:core, :, :], T.ab[:, :, :core, :], optimize=True) # (12)
    X3B[:core, :, :, core:, :] -= np.einsum("imk,bcjm->ibcjk", X["ab"]["oooi/jk"][:core, :, :], T.ab[:, :, core:, :], optimize=True) # (12)
    X3B[core:, :, :, :core, :] -= np.einsum("imk,bcjm->ibcjk", X["ab"]["oooi/jk"][core:, :, :], T.ab[:, :, :core, :], optimize=True) # (12)
    X3B[:core, :, :, :core, :] -= np.einsum("imk,bcjm->ibcjk", X["ab"]["oooI/jk"], T.ab[:, :, :core, :], optimize=True) # (13)
    X3B[:core, :, :, :core, :] += np.einsum("ice,bejk->ibcjk", X["ab"]["ovvi/jk"][:core, :, :], T.ab[:, :, :core, :], optimize=True) # (14)
    X3B[:core, :, :, :core, :] += np.einsum("ice,bejk->ibcjk", X["ab"]["ovvI/jk"], T.ab[:, :, :core, :], optimize=True) # (15)
    X3B[:core, :, :, core:, :] += np.einsum("ice,bejk->ibcjk", X["ab"]["ovvi/jk"][:core, :, :], T.ab[:, :, core:, :], optimize=True) # (14)
    X3B[core:, :, :, :core, :] += np.einsum("ice,bejk->ibcjk", X["ab"]["ovvi/jk"][core:, :, :], T.ab[:, :, :core, :], optimize=True) # (14)
    X3B[:core, :, :, :core, :] += np.einsum("ibe,ecjk->ibcjk", X["aa"]["ovvi/jk"][:core, :, :], T.ab[:, :, :core, :], optimize=True)  # (16)
    X3B[:core, :, :, core:, :] += np.einsum("ibe,ecjk->ibcjk", X["aa"]["ovvi/jk"][:core, :, :], T.ab[:, :, core:, :], optimize=True)  # (16)
    X3B[core:, :, :, :core, :] += np.einsum("ibe,ecjk->ibcjk", X["aa"]["ovvi/jk"][core:, :, :], T.ab[:, :, :core, :], optimize=True)  # (16)
    X3B[:core, :, :, :core, :] += np.einsum("ibe,ecjk->ibcjk", X["aa"]["ovvI/jk"], T.ab[:, :, :core, :], optimize=True)  # (17)
    # parts with T3
    X3B[:core, :, :, :core, :] += np.einsum("iem,ebcmjk->ibcjk", X["aa"]["ovoi/jk"][:core, :, :], T.aab[:, :, :, :, :core, :], optimize=True) # [1]
    X3B[:core, :, :, core:, :] += np.einsum("iem,ebcmjk->ibcjk", X["aa"]["ovoi/jk"][:core, :, :], T.aab[:, :, :, :, core:, :], optimize=True) # [1]
    X3B[core:, :, :, :core, :] += np.einsum("iem,ebcmjk->ibcjk", X["aa"]["ovoi/jk"][core:, :, :], T.aab[:, :, :, :, :core, :], optimize=True) # [1]
    X3B[:core, :, :, :core, :] += np.einsum("iem,ebcmjk->ibcjk", X["aa"]["ovoI/jk"], T.aab[:, :, :, :, :core, :], optimize=True) # [1]
    X3B[:core, :, :, :core, :] += np.einsum("iem,becjmk->ibcjk", X["ab"]["ovoi/jk"][:core, :, :], T.abb[:, :, :, :core, :, :], optimize=True) # [2]
    X3B[:core, :, :, core:, :] += np.einsum("iem,becjmk->ibcjk", X["ab"]["ovoi/jk"][:core, :, :], T.abb[:, :, :, core:, :, :], optimize=True) # [2]
    X3B[core:, :, :, :core, :] += np.einsum("iem,becjmk->ibcjk", X["ab"]["ovoi/jk"][core:, :, :], T.abb[:, :, :, :core, :, :], optimize=True) # [2]
    X3B[:core, :, :, :core, :] += np.einsum("iem,becjmk->ibcjk", X["ab"]["ovoI/jk"], T.abb[:, :, :, :core, :, :], optimize=True) # [2]
    X3B[:core, :, :, :core, :] -= 0.5 * np.einsum("emk,ebcijm->ibcjk", X["ab"]["voo"], T.aab[:, :, :, :core, :core, :], optimize=True) # [3]
    X3B[:core, :, :, core:, :] -= np.einsum("emk,ebcijm->ibcjk", X["ab"]["voo"], T.aab[:, :, :, :core, core:, :], optimize=True) # [3]
    X3B[:core, :, :, :core, :] += 0.25 * np.einsum("bfe,efcijk->ibcjk", X["aa"]["vvv"], T.aab[:, :, :, :core, :core, :], optimize=True) # [4]
    X3B[:core, :, :, core:, :] += 0.5 * np.einsum("bfe,efcijk->ibcjk", X["aa"]["vvv"], T.aab[:, :, :, :core, core:, :], optimize=True) # [4]
    X3B[:core, :, :, :core, :] += 0.5 * np.einsum("ecf,ebfijk->ibcjk", X["ab"]["vvv"], T.aab[:, :, :, :core, :core, :], optimize=True) # [5]
    X3B[:core, :, :, core:, :] += np.einsum("ecf,ebfijk->ibcjk", X["ab"]["vvv"], T.aab[:, :, :, :core, core:, :], optimize=True) # [5]
    X3B[:core, :, :, :core, :] += 0.5 * np.einsum("e,ebcijk->ibcjk", X["a"]["v"], T.aab[:, :, :, :core, :core, :], optimize=True)      # [6]
    X3B[:core, :, :, core:, :] += np.einsum("e,ebcijk->ibcjk", X["a"]["v"], T.aab[:, :, :, :core, core:, :], optimize=True)      # [6]

    X3B -= np.transpose(X3B, (3, 1, 2, 0, 4)) # antisymmetrize (ij)

    tmp = - np.einsum("mj,ibcmk->ibcjk", H.a.oo[core:, core:], Raab["cv"], optimize=True) # (1)
    tmp += np.einsum("mnjk,ibcmn->ibcjk", H.ab.oooo[core:, :, core:, :], Raab["cv"], optimize=True) # (2)
    tmp += np.einsum("bmje,iecmk->ibcjk", H.aa.voov[:, core:, core:, :], Raab["cv"], optimize=True) # (3)
    tmp += np.einsum("bmje,iecmk->ibcjk", H.ab.voov[:, :, core:, :], Rabb["c"], optimize=True) # (4)
    tmp -= np.einsum("mcje,ibemk->ibcjk", H.ab.ovov[core:, :, core:, :], Raab["cv"], optimize=True) # (5)
    # moment-like terms
    tmp -= np.einsum("mcjk,ibm->ibcjk", H.ab.ovoo[core:, :, core:, :], Raa["cv"], optimize=True) # (6)
    tmp -= np.einsum("bmjk,icm->ibcjk", H.ab.vooo[:, :, core:, :], Rab["c"], optimize=True) # (7)
    tmp += np.einsum("bcje,iek->ibcjk", H.ab.vvov[:, :, core:, :], Rab["c"], optimize=True) # (8)
    # 3-body Hbar terms factorized using intermediates
    tmp += np.einsum("ice,bejk->ibcjk", X["ab"]["ovvI/jk"], T.ab[:, :, core:, :], optimize=True) # (15)
    tmp -= np.einsum("imk,bcjm->ibcjk", X["ab"]["oooI/jk"], T.ab[:, :, core:, :], optimize=True) # (13)
    tmp -= np.einsum("imj,bcmk->ibcjk", X["aa"]["oooi_j/k"][:core, :, core:], T.ab, optimize=True) # (11)
    tmp += np.einsum("ibe,ecjk->ibcjk", X["aa"]["ovvI/jk"], T.ab[:, :, core:, :], optimize=True)  # (17)
    # parts with T3
    tmp += np.einsum("iem,ebcmjk->ibcjk", X["aa"]["ovoI/jk"], T.aab[:, :, :, :, core:, :], optimize=True) # [1]
    tmp += np.einsum("iem,becjmk->ibcjk", X["ab"]["ovoI/jk"], T.abb[:, :, :, core:, :, :], optimize=True) # [2]

    X3B[:core, :, :, core:, :] += tmp
    X3B[core:, :, :, :core, :] -= tmp.transpose(3, 1, 2, 0, 4)

    return X3B

def build_HR_3C(R, Rab, Raab, Rabb, T, X, H, core):
    """Calculate the projection <Ij~k~b~c~|[ (H_N e^(T1+T2))_C*(R1h+R2h1p+R3h2p) ]_C|0>."""
    X3C = np.zeros_like(R.abb)

    X3C[:core, :, :, :, :] -= 0.5 * np.einsum("mj,ibcmk->ibcjk", H.b.oo, Rabb["c"], optimize=True) 
    X3C[:core, :, :, :, :] -= 0.25 * np.einsum("mi,mbcjk->ibcjk", H.a.oo[:core, :core], Rabb["c"], optimize=True) 
    X3C[:core, :, :, :, :] += 0.5 * np.einsum("be,iecjk->ibcjk", H.b.vv, Rabb["c"], optimize=True) 
    X3C[:core, :, :, :, :] += (1.0 / 8.0) * np.einsum("mnjk,ibcmn->ibcjk", H.bb.oooo, Rabb["c"], optimize=True) 
    X3C[:core, :, :, :, :] += 0.5 * np.einsum("mnij,mbcnk->ibcjk", H.ab.oooo[:core, :, :core, :], Rabb["c"], optimize=True) 
    X3C[:core, :, :, :, :] += (1.0 / 8.0) * np.einsum("bcef,iefjk->ibcjk", H.bb.vvvv, Rabb["c"], optimize=True) 
    X3C[:core, :, :, :, :] += np.einsum("mbej,iecmk->ibcjk", H.ab.ovvo, Raab["c"], optimize=True) 
    X3C[:core, :, :, :, :] += np.einsum("bmje,iecmk->ibcjk", H.bb.voov, Rabb["c"], optimize=True)
    X3C[:core, :, :, :, :] -= 0.5 * np.einsum("mbie,mecjk->ibcjk", H.ab.ovov[:core, :, :core, :], Rabb["c"], optimize=True)
    # moment-like terms
    X3C[:core, :, :, :, :] -= np.einsum("mcik,mbj->ibcjk", H.ab.ovoo[:core, :, :core, :], Rab["c"], optimize=True) 
    X3C[:core, :, :, :, :] -= 0.5 * np.einsum("cmkj,ibm->ibcjk", H.bb.vooo, Rab["c"], optimize=True) 
    X3C[:core, :, :, :, :] += 0.5 * np.einsum("cbke,iej->ibcjk", H.bb.vvov, Rab["c"], optimize=True) 
    # 3-body Hbar terms factorized using intermediates
    X3C[:core, :, :, :, :] -= 0.5 * np.einsum("imj,bcmk->ibcjk", X["ab"]["oooi/jk"][:core, :, :], T.bb, optimize=True) 
    X3C[:core, :, :, :, :] -= 0.5 * np.einsum("imj,bcmk->ibcjk", X["ab"]["oooI/jk"], T.bb, optimize=True) 
    X3C[:core, :, :, :, :] += 0.5 * np.einsum("ibe,ecjk->ibcjk", X["ab"]["ovvi/jk"][:core, :, :], T.bb, optimize=True) 
    X3C[:core, :, :, :, :] += 0.5 * np.einsum("ibe,ecjk->ibcjk", X["ab"]["ovvI/jk"], T.bb, optimize=True) 
    X3C[:core, :, :, :, :] += np.einsum("ebj,ecik->ibcjk", X["ab"]["vvo"], T.ab[:, :, :core, :], optimize=True) 
    # parts with T3
    X3C[:core, :, :, :, :] += (1.0 / 4.0) * np.einsum("iem,ebcmjk->ibcjk", X["aa"]["ovoi/jk"][:core, :, :], T.abb, optimize=True) 
    X3C[:core, :, :, :, :] += (1.0 / 4.0) * np.einsum("iem,ebcmjk->ibcjk", X["aa"]["ovoI/jk"], T.abb, optimize=True) 
    X3C[:core, :, :, :, :] += (1.0 / 4.0) * np.einsum("iem,ebcmjk->ibcjk", X["ab"]["ovoi/jk"][:core, :, :], T.bbb, optimize=True)
    X3C[:core, :, :, :, :] += (1.0 / 4.0) * np.einsum("iem,ebcmjk->ibcjk", X["ab"]["ovoI/jk"], T.bbb, optimize=True) 
    X3C[:core, :, :, :, :] -= (2.0 / 4.0) * np.einsum("emj,ebcimk->ibcjk", X["ab"]["voo"], T.abb[:, :, :, :core, :, :], optimize=True) 
    X3C[:core, :, :, :, :] += (2.0 / 4.0) * np.einsum("ebf,efcijk->ibcjk", X["ab"]["vvv"], T.abb[:, :, :, :core, :, :], optimize=True) 
    X3C[:core, :, :, :, :] += (1.0 / 4.0) * np.einsum("e,ebcijk->ibcjk", X["a"]["v"], T.abb[:, :, :, :core, :, :], optimize=True)

    X3C -= np.transpose(X3C, (0, 2, 1, 3, 4)) # antisymmetrize A(bc)
    X3C -= np.transpose(X3C, (0, 1, 2, 4, 3)) # antisymmetrize A(jk)

    return X3C
