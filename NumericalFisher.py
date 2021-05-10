import numpy as np
import sklearn.decomposition as skdecomp
import matplotlib
import matplotlib.pyplot as plt

class Fisher():
    def __init__(self,fid,derivs_m,derivs_p,offsets,pca_std_cut=None):
        self.n_fid=fid.shape[0]
        self.stat_len=fid.shape[1]
        self.n_pars=len(derivs_p)
        self.mean_fid,self.std_fid=np.mean(fid,axis=0),np.std(fid,axis=0)
        self.fid_n=(fid-self.mean_fid[None,:])/self.std_fid[None,:]
        
        self.ms_n=(derivs_m-self.mean_fid[None,None,:])/self.std_fid[None,None,:]
        self.ps_n=(derivs_p-self.mean_fid[None,None,:])/self.std_fid[None,None,:]
        self.offsets=offsets
        self.n_derivs=self.ms_n[0].shape[0]
        
        if pca_std_cut is not None:
            self.fit_fiducial_pca()
            self._valid=self.pca_std>pca_std_cut
            
    @property
    def pca(self):
        try:
            return self._pca
        except:
            self.fit_fiducial_pca()
            return self._pca
        
    @property
    def valid(self):
        try:
            return self._valid
        except:
            self.pca
            print("Set valid pca_std cut")
            self.see_pca_stds()
            while True:
                try:
                    std_cut=float(input("std_cut "))
                    break
                except Exception as exp:
                    print(exp)
                    print("Invalid")
            self._valid=self.pca_std>std_cut
            return self._valid
    
    def fit_fiducial_pca(self):
        print("Fitting PCA")
        self._pca=skdecomp.PCA()
        self._pca.fit(self.fid_n)#fit transfrom results is slightly off from transform
        pca_res=self._pca.transform(self.fid_n)
        self.pca_std=np.std(pca_res,axis=0)
    
    def see_pca_stds(self):
        plt.figure(figsize=(5,4))
        _=plt.hist(self.pca_std,bins=np.logspace(np.log10(np.min(self.pca_std)*0.9+1e-8),np.log10(np.max(self.pca_std)*1.1),100,base=10))
        plt.yscale("log")
        plt.xscale("log")
        plt.show()
    
    def get_derivs(self,n=None,pca=False):
        derivs=[]
        for m_n,p_n,offset in zip(self.ms_n,self.ps_n,self.offsets):
            if n is not None:
                choices=np.random.choice(self.n_derivs,n,replace=False)
                m_n=m_n[choices]
                p_n=p_n[choices]
            deriv=(p_n.mean(0)-m_n.mean(0))/(2*offset)
            if pca:
                deriv=self.pca.transform(deriv[None,:])[0,self.valid]
            derivs.append(deriv)
        return np.array(derivs)
    
    def get_cov(self,n=None,pca=False):
        fid_n=self.fid_n.copy()
        if n is not None:
            fid_n=fid_n[np.random.choice(self.n_fid,n,replace=False)]
        if not pca:
            return np.cov(fid_n.T)
        pca_res=self.pca.transform(fid_n)
        pca_n=pca_res[:,self.valid]
        return np.cov(pca_n.T)

    def get_cov_par(self,n_fid_use=None,n_derivs_use=None,pca=False,full_output=False):
        cov=self.get_cov(n_fid_use,pca)
        fish=np.linalg.inv(cov)
        derivs=self.get_derivs(n_derivs_use,pca)
        fish_par=derivs@fish@derivs.T
        cov_par=np.linalg.inv(fish_par)
        if full_output:
            return {"cov_par":cov_par,"fish_par":fish_par,"cov":cov,"fish":fish,"derivs":derivs}
        return cov_par
    
    def get_convergence(self,key,evaluate_which,n_init=None,n_bins=50,pca=False):
        assert key in ["cov_par","fish_par","cov","fish","derivs"], "key should be in"+str(["cov_par","fish_par","cov","fish","derivs"])
        assert evaluate_which in ["fid","derivs"],"evaluate_which should be in"+str(["fid","derivs"])
        
        n_tot=self.n_fid if evaluate_which=="fid" else self.n_derivs
        if n_init is None:
            n_init=n_tot//4
        ns=np.linspace(n_init,n_tot,n_bins).astype(np.int64)
        conv=[]
        for n in ns:
            if evaluate_which=="fid":
                conv.append(self.get_cov_par(n_fid_use=n,full_output=True,pca=pca)[key])
            else:
                conv.append(self.get_cov_par(n_derivs_use=n,full_output=True,pca=pca)[key])
        conv=np.array(conv).reshape(n_bins,-1)
        return ns,conv
    
    def plot_convergence(self,ns,conv,key="",ax=None):
        conv_rel=np.abs(conv/conv[[-1],:]-1)
        if ax is None:
            fig=plt.figure(figsize=(7,5))
            ax=fig.add_subplot(1,1,1)
        for i in range(conv_rel.shape[1]):
            ax.plot(ns,conv_rel[:,i])
        
        ax.set_yscale("log")
        ax.set_title(key+" Convergence")
        ax.set_ylabel("abs(val/val_n_max-1)")
        ax.set_xlabel("n")
        

    

def plot(means,covs,labels=None,par_labels=None,colors=None,lws=None,fig=None,alpha=1.52,vis=1.4):
    from matplotlib.patches import Ellipse
    import matplotlib.transforms as transforms
    
    n_pars=covs[0].shape[0]

    if fig is None:
        fig=plt.figure(figsize=(15,15))
    subplots={}
    for i in range(n_pars):
        for j in range(n_pars):
            if j>i:
                continue
            ax=fig.add_subplot(n_pars,n_pars,n_pars*i+j+1)
            subplots[(i,j)]=ax
            if j==0:
                ax.set_ylabel(par_labels[i])
            if i==n_pars-1:
                ax.set_xlabel(par_labels[j])
            if i==j:
                for cov_count,cov in enumerate(covs):
                    var=cov[i,i]
                    xs=np.linspace(means[i]-10*np.sqrt(var),means[i]+10*np.sqrt(var),1000)
                    ax.plot(xs,np.exp(-(xs-means[i])**2/(2*var))/np.sqrt(2*np.pi*var),lw=1 if lws is None else lws[cov_count], c=None if colors is None else colors[cov_count])
                continue
            for cov_count,cov in enumerate(covs):
                subcov=cov[[j,i]][:,[j,i]]
                sumvar=subcov[0, 0]+subcov[1, 1]
                diff=subcov[0, 0]-subcov[1, 1]
                corrsq=subcov[0,1]**2
                a=np.sqrt(sumvar/2+np.sqrt(diff**2/4+corrsq))
                b=np.sqrt(sumvar/2-np.sqrt(diff**2/4+corrsq))
                theta=np.arctan2(a**2-subcov[0,0],subcov[0,1])
                ellipse = Ellipse((0, 0), width=2*a*alpha, height=2*b*alpha,facecolor="none",lw=1 if lws is None else lws[cov_count], edgecolor=None if colors is None else colors[cov_count])
                transf = transforms.Affine2D().rotate(theta).translate(means[j], means[i])
                ellipse.set_transform(transf + ax.transData)
                ax.add_patch(ellipse)
    lims=[np.array([np.inf,-np.inf]) for _ in range(len(means))]
    for i in range(n_pars):
        for j in range(n_pars):
            if j>=i:
                continue
            ax=subplots[(i,j)]
            ax.autoscale_view(tight=False)
            mx,Mx=ax.get_xlim()
            my,My=ax.get_ylim()
            lims[j][0]=min(lims[j][0],mx)
            lims[j][1]=max(lims[j][1],Mx)
            lims[i][0]=min(lims[i][0],my)
            lims[i][1]=max(lims[i][1],My)
    lims=np.array(lims)-np.array(means)[:,None]
    lims=np.abs(lims).max(axis=1)*vis
    for i in range(n_pars):
        for j in range(n_pars):
            if j>i:
                continue
            ax=subplots[(i,j)]
            ax.set_xlim(means[j]-lims[j],means[j]+lims[j])
            if i==j:
                continue
            ax.set_ylim(means[i]-lims[i],means[i]+lims[i])
    if labels is not None:
        lines=[]
        for cov_count in range(len(labels)):
            lines.extend(plt.plot(0,0,c=None if colors is None else colors[cov_count],lw=1 if lws is None else lws[cov_count]))
        fig.legend(lines,labels)
    return fig


    
    
    
    
    
    
    
    
    
    
    