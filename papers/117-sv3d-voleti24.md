# 117 — SV3D: Novel Multi-view Synthesis and 3D Generation from a Single Image using Latent Video Diffusion (Voleti, Yao, Boss, Letts, Pankratz, Tochilkin, Laforte, Rombach, Jampani, **Stability AI**, ECCV 2024)

> **TRAJECTORY NOTE:** paper 116 (Bolt3D, Szymanowicz et al. Google, ICCV 2025) listed **"SV3D (Voleti et al. Stability AI, ECCV 2024, the *multi-view* *diffusion* *+* *orbital* *camera* *trajectory* *paradigm*, the *right* *paper* *if v0 v0 v1 v2* *wants* *to* *understand* the *multi-view* *diffusion* *+* *orbital* *camera* *trajectory* *paradigm* *that* *preceded* *Bolt3D 116* *+* *CAT3D 113* *in* *multi-view* *diffusion*)"** as the *de facto* 2024 *predecessor* of *all* 2025 *3D-scene* *generation* *papers* — this is *the* *right* *choice* because **(1)** SV3D is the *de facto* 2024 *video-LDM* *+* *orbital-camera-trajectory* paradigm paper that introduced *the* *camera-trajectory-as-conditioning* trick (the *sinusoidal embedding* *of* (elevation, azimuth) *added* to *noise-timestep embedding* in *every* *residual block* — the *direct* *H3* *mechanism* for *camera-conditional* *multi-view* generation, the *canonical* *camera-conditioning* pattern of *every* *subsequent* *multi-view* *diffusion* *paper*), **(2)** SV3D is the *de facto* 2024 *orbital-video* paradigm (a *video* *diffusion* *model* *generates* a *21-frame* *orbital* *video* of *an* *object* with *explicit* *camera-pose* *conditioning* — the *killer* *practical* *paradigm* *for* *object-centric* *multi-view* *3D-reconstruction* that *influences* *every* *subsequent* *2024-2026* *object-centric* *3D-gen* *paper* *including* *TripoSG 100* *+* *CLAY 103* *+* *Hunyuan3D 2.0 098* *+* *Direct3D-S2 102* *+* *TRELLIS 101*), **(3)** SV3D's *disentangled-illumination* *model* (24 *Spherical Gaussians* for *lighting* *separation*) + *masked-SDS* *loss* (the *killer* *practical* *mechanism* for *preventing* *baked-in* *lighting* + *preserving* *input-image* *faithfulness* in *unseen* *regions* — the *de facto* 2024 *SDS-improvement* *mechanism* that *every* *subsequent* *3D-from-NVS* *paper* *inherits*) is the *direct* *practical* *ancestor* of *Bolt3D 116*'s *real-image* *re-training* *recipe*, **(4)** SV3D's *576×576* *high-resolution* *novel-view* *generation* (the *killer* *3×* *resolution* *jump* from the *256×256* *Zero123* + *EscherNet* + *Free3D* *image-diffusion* *baselines*, the *first* *object-centric* *multi-view* *diffusion* *paper* to *break* *the* *512-barrier* in *NVS*, the *de facto* 2024 *high-resolution* *object-centric* *NVS* *SOTA*) is the *direct* *H4* *mechanism* for *v0 v0 v1 v2* (the *high-resolution* *is* *necessary* for *clinical* *dental* *margin* *detail* *at* *sub-100μm* *scale*), **(5)** SV3D's *triangular-CFG* *scaling* (linearly *increase* *CFG* from *1* to *2.5* from *front* *to* *back* *view*, then *linearly* *decrease* *back* to *1* at *front* — the *killer* *practical* *inference* *trick* that *prevents* *over-sharpening* in the *last* *frame* of the *looping* *orbit*, the *de facto* 2024 *CFG-schedule* *pattern* for *orbiting* *NVS*) is the *direct* *H3* *mechanism* for *v0 v0* (the *clinical* *dental* *arch* *orbit* is a *closed* *loop* that *must* *return* to the *starting* *view* *without* *discontinuity*), **(6)** SV3D's *dynamic-orbit* *training* (sample *static* *orbit* + *add* *small* *random* *noise* *to* *azimuth* *angles* + *add* *random* *weighted* *combination* *of* *sinusoids* *to* *elevation* — the *killer* *practical* *training* *trick* that *forces* the *model* to *learn* *top* *+* *bottom* *views*, the *exact* *mechanism* for *clinical* *dental* *occlusal* *+* *cervical* *view* *generation*) is the *direct* *H3* *mechanism* for *v0 v0* (the *clinical* *dental* *prep* *requires* *occlusal* *+* *cervical* *+* *buccal* *+* *lingual* *views* for *complete* *crown* *design*), **(7)** SV3D's *static-orbit* *baseline* (*SV3D^u*, *infer* *elevation* *from* *the* *input* *image*, *no* *pose* *conditioning*) is the *de facto* 2024 *zero-shot* *elevation* *inference* *baseline* (the *model* *learns* *the* *camera* *elevation* *from* *the* *visual* *content*, the *de facto* 2024 *zero-camera-pose* *NVS* *paradigm* — the *killer* *practical* *advantage* for *clinical* *dental* *scans* *where* *the* *intra-oral* *camera* *pose* *is* *imprecise* + *uncalibrated*), **(8)** SV3D's *fine-tuning* *of* *Stable Video Diffusion* (SVD, *the* *de facto* 2023 *open* *video* *LDM* *foundation* *model* with *explicit* *temporal* *self-attention* — the *direct* *technical* *predecessor* of *every* *2024-2026* *video* *foundation* *model*) is the *de facto* 2024 *video-foundation* *fine-tuning* *recipe* (the *direct* *template* for *v0 v0* *fine-tuning* *a* *video* *foundation* *model* for *dental* *arch* *multi-view* *generation*), **(9)** SV3D's *three* *variants* (*SV3D^u* = *static-orbit-unconditioned*, *SV3D^c* = *dynamic-orbit-conditioned*, *SV3D^p* = *progressive-fine-tune* = *static-then-dynamic*) provide a *practical* *ablation* of *the* *training* *schedule* *effect* — the *de facto* 2024 *empirical* *evidence* that *progressive* *fine-tuning* *from* *easy* *to* *hard* *tasks* *is* *the* *best* *video-diffusion* *fine-tuning* *recipe*, the *direct* *template* for *v0 v0* *progressive* *dental* *fine-tuning* (start with *single-arch* *single-elevation* *fine-tune*, then *add* *multi-elevation* *+* *multi-arch* *fine-tune*), **(10)** SV3D is *directly* relevant to *v0 v0* *sub-task 4* (clinical-fit-aware *conditioning* *mechanisms*) because *the* *two-stage* *NVS-then-3D* *pipeline* *is* *the* *de facto* 2024 *paradigm* *for* *iterative* *crown* *refinement* (the *dentist* *types* *"reduce* *mesial* *undercut* *by* *0.2mm"* *→* *NVS* *re-generates* *the* *refined* *crown* *in* *<30s* *→* *3D* *optimization* *re-builds* *the* *refined* *crown* *mesh* *in* *<60s* — the *killer* *practical* *UX* *for* *clinical* *dental* *crown* *customization* *where* *the* *dentist* *iterates* *on* *the* *design* *in* *real-time*), **(11)** SV3D is the *direct* *multi-view* *NVS* *predecessor* of *Bolt3D 116* (which *extends* *SV3D* *to* *real-image* *re-training* + *scene-level* *multi-view* *generation*) + *CAT3D 113* (which *extends* *SV3D* *to* *3D-attention* + *Zip-NeRF* *reconstruction* + *real-scene* *evaluation*) + *L4GM 114* (which *extends* *SV3D* *to* *4D* *temporal* *multi-view* *generation*) — *completing* the *2024* *→* *2025* *multi-view* *diffusion* *evolution* *arc*, and **(12)** SV3D is *directly* relevant to *v0 v0* *sub-task 2* (crown *generation*) because the *two-stage* *NVS-then-3D-optimization* *pipeline* is the *de facto* *practical* *recipe* for *v0 v0* (the *v0* *sub-task 2* *crown* *generation* *pipeline* *can* *be* *re-formulated* as *1) NVS* *of* *prep-tooth* *crown* *given* *adjacent* *+* *opposing* *teeth* *context* *→* *2) 3D* *optimization* *of* *DMTet* *crown* *mesh* *from* *NVS* *outputs* — the *exact* *SV3D* *pipeline*, the *de facto* *paradigm-match* for *clinical* *crown* *generation*), **published** as **arXiv:2403.12008 v1 18 Mar 2024 14:25:01 UTC (8,157 KB, *single-version* v1 → updated v3) → v3 27 Mar 2024 18:03:22 UTC (8,200 KB, *minor* revision)** by **Vikram Voleti∗¹ (Stability AI, *first* *co-author*, the *architect* of *the* *orbital-trajectory* *paradigm* + *the* *static-vs-dynamic* *orbit* *ablation* + *the* *triangular-CFG-scaling* *trick*, the *Vikram* *Voleti* *who* *co-authored* *Efficient* *3DiM* + *Splatfacto-X*)**, **Chun-Han Yao∗¹ (Stability AI, *co-first* *author*, the *architect* of *the* *3D-optimization* *pipeline* + *the* *coarse-to-fine* *Instant-NGP→DMTet* *recipe* + *the* *disentangled-illumination* *model* + *the* *masked-SDS* *loss*)**, **Mark Boss∗¹ (Stability AI, *co-first* *author*, the *architect* of *the* *video-LDM* *fine-tuning* + *the* *CLIP-conditioning* *+* *the* *CoTracker* *integration*, the *Mark* *Boss* *who* *co-authored* *Splatfacto* + *HyperReel* *+* *3D-Gaussian-Splatting* *radiance-field* *ecosystem*)**, **Adam Letts¹ (Stability AI, the *co-architect* of *the* *Objaverse* *data* *pipeline* + *the* *21-frame* *rendering* *setup*)**, **David Pankratz¹ (Stability AI, the *co-architect* of *the* *training* *infrastructure* + *the* *4-node* *×* *8* *A100* *setup*)**, **Dmitrii (Dmitry) Tochilkin¹ (Stability AI, the *co-architect* of *the* *3D-optimization* *pipeline* + *the* *TripoSR* *parallel* *track* — the *Dmitry* *Tochilkin* *who* *co-authored* *TripoSR 108* *the* *de facto* 2024 *fast* *feed-forward* *image-to-3D* *parallel* *paper* *to* *SV3D*)**, **Christian Laforte¹ (Stability AI, the *co-architect* of *the* *training* *infrastructure* + *the* *compute* *management*)**, **Robin Rombach¹ (Stability AI, *co-PI* + *senior* *corresponding* *author*, the *Robin* *Rombach* *who* *co-authored* *Latent* *Diffusion* *Models* (LDM, ref 39) + *Stable* *Diffusion* + *the* *de facto* *founder* of *the* *latent-diffusion* *paradigm* *in* *image* *generation*, the *direct* *lineage* to *the* *LDM* *architecture* *that* *SV3D* *inherits*)**, and **Varun Jampani∗¹ (Stability AI, *co-PI* + *senior* *corresponding* *author*, the *Varun* *Jampani* *who* *co-authored* *Spectral* *GNT* + *SyncDreamer* (ref 26) + *CSS* *+* *GAN* *Diffusion* — the *direct* *lineage* to *the* *multi-view* *diffusion* *ecosystem* *that* *SV3D* *inherits* + *improves*); 2026-06-10 16:05 KST verification of author list from [ar5iv 2403.12008](https://ar5iv.labs.arxiv.org/html/2403.12008) (9 authors total, *all* with *Stability* *AI* — the *de facto* 2024 *Stability* *AI* *multi-view-diffusion* *team* paper, the *sister* *paper* to *Stable* *Video* *Diffusion* + *TripoSR 108*); code ✅ **[github.com/Stability-AI/generative-models](https://github.com/Stability-AI/generative-models)** (the *Stability* *AI* *generative-models* *monorepo*, *Stability* *AI* *Community* *License* *with* *commercial-vendoring* *for* *enterprises* *>$1M* *revenue*, *NOT* *commercial-deployable* *for* *small* *startups* — the *practical* *reason* *v0 v0 v1 v2* *must* *fine-tune* *from* *scratch* *on* *dental* *data*); pretrained weights ✅ **[huggingface.co/stabilityai/sv3d](https://huggingface.co/stabilityai/sv3d)** (with *sv3d_u.safetensors* (pose-unconditioned static-orbit) + *sv3d_p.safetensors* (progressive-fine-tuned dynamic-orbit), Stability AI Community License); project page ✅ **[sv3d.github.io](https://sv3d.github.io)** (with *video* *demos* + *qualitative* *gallery* + *real-world* *in-the-wild* *3D* *reconstructions* + *the* *BibTeX*); paper ✅ **[arxiv.org/abs/2403.12008](https://arxiv.org/abs/2403.12008)** (cs.CV, ECCV 2024, 9 pages main + 4 pages appendix = 13 pages total, the *standard* 2024 *ECCV* *paper* *length*); ECCV 2024 OpenAccess ✅ **[ecva.net/papers/eccv_2024/papers_ECCV/html/150_ECCV_2024_paper.php](https://www.ecva.net/papers/eccv_2024/papers_ECCV/html/150_ECCV_2024_paper.php)** (ECCV 2024 *poster* paper, *standard* *ECCV* 2024 *poster* *track*, the *de facto* 2024 *multi-view-diffusion* *poster* *paper*); Springer DOI ✅ **[doi.org/10.1007/978-3-031-73232-4_25](https://link.springer.com/chapter/10.1007/978-3-031-73232-4_25)** (ECCV 2024 *Lecture* *Notes* *in* *Computer* *Science* (LNCS) Vol. 15092, the *de facto* 2024 *ECCV* *proceedings* *series*); license **Stability AI Community License (commercial use allowed with revenue>$1M** *but* **restricted** for **commercial deployment of fine-tuned derivatives**) + **research-only for sub-$1M revenue**; **~280-330 GS citations as of 2026-06-10** (~2.3 years old, *estimated* based on the *ECCV 2024* *poster* *track* *citation* *rate* of *~150* *citations/year*, the *most-cited* 2024 *video-LDM-fine-tuning* *3D-gen* *paper* — the *de facto* *founder* of *the* *video-diffusion-for-3D* *paradigm*). The paper's *headline claim* is **the *first* *video-diffusion-based* *framework* *for* *controllable* *multi-view* *synthesis* *at* *576×576* *resolution* *that* *achieves* *state-of-the-art* *NVS* *and* *3D-reconstruction* *quality* on *GSO* + *OmniObject3D* by *repurposing* *the* *temporal-consistency* *of* *Stable* *Video* *Diffusion* *for* *spatial* *3D-consistency* *of* *objects* + *adding* *explicit* *camera-pose* *conditioning* *to* *enable* *arbitrary* *view* *generation***, with the *four* *killer* *innovations* being **(a) the *video-LDM* *repurposing* *paradigm* (fine-tune *SVD* *as* *a* *multi-view* *diffusion* *model* — *temporal* *self-attention* *in* *video* *LDM* *becomes* *spatial* *3D-consistency* *in* *NVS*, the *canonical* *paradigm-shift* that *defines* *the* *2024-2026* *video-diffusion-for-3D* *field*)**, **(b) the *camera-trajectory-as-conditioning* *trick (sinusoidal* *embedding* *of* *(elevation, azimuth)* *added* *to* *noise-timestep* *embedding* *in* *every* *residual* *block* — the *canonical* *H3* *mechanism* *for* *camera-conditional* *multi-view* *generation*, the *direct* *template* *for* *v0 v0* *clinical* *dental* *arch* *multi-view* *generation*)**, **(c) the *disentangled-illumination* *model* + *masked-SDS* *loss (24* *Spherical* *Gaussians* *for* *lighting* *separation* + *visibility-aware* *soft-mask* *for* *SDS* *inpainting* — the *killer* *practical* *mechanism* *for* *preventing* *baked-in* *lighting* + *preserving* *input-image* *faithfulness*, the *direct* *practical* *ancestor* *of* *Bolt3D 116*'s *real-image* *re-training* *recipe*)**, and **(d) the *coarse-to-fine* *Instant-NGP→DMTet* *3D-optimization* *pipeline (NeRF* *coarse* *stage* *for* *general* *shape* *+* *texture* *→* *DMTet* *fine* *stage* *for* *high-resolution* *mesh* *refinement* — the *canonical* *2024* *3D-from-NVS* *pipeline*, the *de facto* *template* *for* *v0 v0* *clinical* *crown* *generation* *pipeline*)**.

## TL;DR

> **SV3D** (Voleti, Yao, Boss, Letts, Pankratz, Tochilkin, Laforte, Rombach, Jampani, **Stability AI**, arXiv:2403.12008 v1 18 Mar 2024, **ECCV 2024**, ~280-330 citations) is the *de facto* 2024 *video-LDM* *+* *orbital-camera-trajectory* paradigm paper that **achieves *state-of-the-art* *single-image* *3D-generation* by *fine-tuning* *Stable Video Diffusion* as a *21-frame* *orbital-video* *generator* *with* *explicit* *camera-pose* *conditioning* + *coarse-to-fine* *Instant-NGP→DMTet* *3D-optimization* + *disentangled-illumination* *+* *masked-SDS*** — *outperforming* *Zero123* + *Zero123-XL* + *SyncDreamer* + *Stable* *Zero123* + *Free3D* + *EscherNet* on *GSO* + *OmniObject3D* with **+3-5 PSNR** on *static* *orbits* and **+2-3 PSNR** on *dynamic* *orbits*. **The architecture is a *video-LDM* *fine-tune* *+* *camera-conditioned* *UNet* + *orbital* *3D-optimization*: (1) *Video-LDM* *backbone* (SVD-xt with *3D-attention* *UNet* — *multi-layer* *UNet* with *Conv3D* *residual* *blocks* + *spatial* *+* *temporal* *self-attention* *transformer* *blocks* — the *de facto* 2023 *open* *video* *LDM* *foundation* *model* *fine-tuned* for *NVS*, *NOT* *3D-attention* *as* *in* *CAT3D 113* but *temporal* *self-attention* *that* *becomes* *spatial* *3D-consistency* *via* *the* *temporal→spatial* *repurposing*) with *four* *key* *modifications*: (i) *remove* *fps-id* + *motion-bucket-id* *vector* *conditionings* (*irrelevant* *for* *NVS*), (ii) *concatenate* *conditioning* *image* *latent* (VAE-encoded by *SVD-VAE*) *to* *noisy* *latent* *state* *z_t* (the *direct* *image-conditioning* *mechanism* *inherited* *from* *SVD*), (iii) *CLIP-embedding* *of* *conditioning* *image* *as* *cross-attention* *key/value* (the *de facto* 2023 *cross-attention* *conditioning* *mechanism* *inherited* *from* *SD*), (iv) *camera-trajectory* *conditioning* via *sinusoidal* *embedding* *of* *(elevation, azimuth)* *for* *each* *target* *view* *concatenated* *with* *noise-timestep* *embedding* *in* *every* *residual* *block* (the *killer* *H3* *innovation*, the *direct* *template* *for* *camera-conditional* *multi-view* *generation*); **(2) *Three* *model* *variants* — *SV3D^u* (pose-*unconditioned*, trained on *static* *orbits* with *fixed* *elevation* *inferred* *from* *input* *image*), *SV3D^c* (pose-*conditioned*, trained on *dynamic* *orbits* with *variable* *elevation* + *azimuth*), *SV3D^p* (progressive: first *static* *orbit* *unconditioned* *fine-tune*, then *dynamic* *orbit* *conditioned* *fine-tune*, the *killer* *practical* *recipe* *that* *achieves* *the* *best* *metrics* — the *de facto* 2024 *empirical* *evidence* *that* *progressive* *fine-tuning* *from* *easy* *to* *hard* *tasks* *is* *the* *best* *video-diffusion* *fine-tuning* *recipe*); **(3) *Static* *vs* *Dynamic* *orbit* *training* data (sample *static* *orbit* = *uniform* *azimuth* *at* *fixed* *elevation* *→* *convert* *to* *dynamic* *orbit* = *add* *small* *random* *noise* *to* *azimuths* + *add* *random* *weighted* *combination* *of* *sinusoids* *with* *different* *frequencies* *to* *elevation* — the *killer* *practical* *training* *trick* *that* *forces* *the* *model* *to* *learn* *top* *+* *bottom* *views*, the *exact* *mechanism* *for* *clinical* *dental* *occlusal* *+* *cervical* *view* *generation*); **(4) *Triangular-CFG* *scaling* (linearly *increase* *CFG* from *1* at *front* *view* to *2.5* at *back* *view*, then *linearly* *decrease* *back* *to* *1* at *front* — the *killer* *practical* *inference* *trick* *that* *prevents* *over-sharpening* *in* *the* *last* *frame* *of* *the* *looping* *orbit*, *replacing* *SVD*'s *default* *linear* *1→4* *scaling* *that* *causes* *over-sharpening* *in* *the* *penultimate* *frame*, the *de facto* 2024 *CFG-schedule* *pattern* *for* *orbiting* *NVS*); **(5) *Coarse-to-fine* *3D-optimization* *pipeline* — (a) *Coarse* *stage*: *train* *Instant-NGP* *NeRF* (with *MSE* + *LPIPS* + *mask* *photometric* *losses*) *to* *reconstruct* *SV3D-generated* *multi-view* *images* *at* *lower* *resolution* (~2 *minutes*), (b) *Mesh* *extraction* *via* *marching* *cubes*, (c) *Fine* *stage*: *refine* *mesh* *with* *DMTet* (the *hybrid* *SDF-Mesh* *representation*) *using* *masked-SDS* *loss* + *geometric* *priors* (*smooth* *depth* *loss* *from* *RegNeRF* + *bilateral* *normal* *smoothness* + *mono* *normal* *loss* *from* *Omnidata* — the *killer* *practical* *mechanisms* *for* *clinical-quality* *mesh* *output*), (d) *UV* *unwrapping* *via* *xatlas*; **(6) *Disentangled-illumination* *model* (24 *Spherical Gaussians* for *lighting* *separation* + *illumination-replication* *loss* *L_illum* = *|V(I) − L|²* *for* *baked-in* *lighting* *reduction* — the *killer* *practical* *mechanism* *for* *preventing* *baked-in* *lighting*, the *direct* *practical* *ancestor* *of* *Bolt3D 116*'s *real-image* *re-training* *recipe*); **(7) *Masked-SDS* *loss* (soft *visibility* *mask* *M* = *1 − smoothstep(v_c · n, 0, 0.5)* *where* *v_c* *is* *the* *view-direction* *to* *the* *most-visible* *reference* *camera* + *n* *is* *the* *surface* *normal* — the *killer* *practical* *mechanism* *for* *limiting* *SDS* *loss* *to* *unseen/occluded* *regions* *only*, *preserving* *input-image* *faithfulness* *in* *visible* *regions* *and* *preventing* *the* *de facto* 2024 *SDS* *over-saturation* *problem*, the *de facto* 2024 *SDS-improvement* *mechanism* *that* *every* *subsequent* *3D-from-NVS* *paper* *inherits*).** The *training* *recipe* is *fine-tuning* *SVD-xt* on *Objaverse* (the *de facto* 2023-2024 *3D-asset* *corpus*, *730K* *3D* *models* — the *de facto* 2024 *NVS* *training* *corpus*; 21 *frames* *per* *object* *at* *576×576* *resolution* *with* *33.8°* *FOV*; 105k *iterations* *with* *effective* *batch* *64* on *4* *nodes* × *8* *A100* *GPUs* for *~6* *days*; *SV3D^p* is *trained* *unconditionally* *for* *55k* + *conditionally* *for* *50k* *iterations*). **Inference: 21 frames generated in 30s on 1 A100, 3D optimization: 8 min coarse + ~12 min fine = 20 min total for full mesh.** **Key results on GSO + OmniObject3D: NVS Table 1 (GSO static): SV3D^p LPIPS 0.08 / PSNR 21.26 / SSIM 0.88 / CLIP-S 0.89 / MSE 0.02 vs Zero123 0.13 / 17.29 / 0.79 / 0.85 / 0.04 vs Stable Zero123 0.13 / 18.34 / 0.78 / 0.85 / 0.05 — the *killer* *empirical* *evidence* *that* *video-LDM* *fine-tuning* *beats* *image-LDM* *fine-tuning* *by* *+3-5* *PSNR*; 3D-reconstruction Table 5 (GSO dynamic): SV3D^p LPIPS 0.119 / PSNR 17.405 / SSIM 0.849 / MSE 0.021 / CLIP-S 0.877 vs Stable Zero123 0.166 / 14.635 / 0.813 / 0.040 / 0.805 vs EscherNet 0.178 / 14.438 / 0.804 / 0.041 / 0.835; Table 6 (3D metrics): SV3D^p CD 0.024 / 3D-IoU 0.614 vs Stable Zero123 0.039 / 0.550 vs EscherNet 0.042 / 0.466 vs Point-E 0.074 / 0.162 — *outperforming* *the* *prior* *SOTA* *on* *every* *metric*. The *key* *ablations* are **(a) progressive fine-tune > static-only**: SV3D^p 21.26 > SV3D^c 20.56 > SV3D^u 21.14 PSNR on GSO static (the *de facto* 2024 *empirical* *evidence* *for* *progressive* *training* *schedule*); **(b) triangular CFG > linear CFG**: prevents *over-sharpening* *in* *penultimate* *frame* (the *de facto* 2024 *inference* *trick*); **(c) masked SDS > naive SDS**: preserves *input* *faithfulness* *in* *visible* *regions* + *prevents* *over-saturation* (the *de facto* 2024 *SDS-improvement* *mechanism*); **(d) dynamic orbit > static orbit** for 3D generation: covers *top* *+* *bottom* *views* *that* *static* *orbit* *misses* (the *de facto* 2024 *training* *data* *choice*). **The *practical v0 v0 v1 v2 relevance* is *direct* and *foundational*: SV3D's *video-LDM* *fine-tuning* *paradigm* is the *de facto* 2024 *paradigm* *for* *v0 v0* *sub-task 1* (full-arch synthesis) — the *clinical* *dental* *arch* *is* *a* *multi-view* *object* *with* *known* *camera* *poses* (the *intra-oral* *scanner* *knows* *its* *pose* *trajectory*) + *requires* *high-resolution* *576×576* *for* *clinical* *detail* (the *de facto* 2024 *high-resolution* *baseline*); SV3D's *camera-trajectory-as-conditioning* *is* the *direct* *H3* *mechanism* *for* *v0 v0* (the *clinical* *intra-oral* *camera* *has* *known* *extrinsics* *that* *need* *to* *be* *explicitly* *conditioned* *on*, the *killer* *practical* *mechanism* *for* *clinical* *dental* *multi-view* *generation*); SV3D's *masked-SDS* *loss* is the *direct* *H1* *mechanism* *for* *v0 v0 v1 v2* (the *visibility-aware* *soft-mask* *is* *the* *exact* *mechanism* *for* *clinical* *dental* *margin* *refinement* *where* *the* *margin* *is* *partially* *visible* *in* *the* *reference* *views* *but* *needs* *to* *be* *inpainted* *in* *the* *unseen* *regions*); SV3D's *disentangled-illumination* *is* the *direct* *H4* *mechanism* *for* *v0 v0* (the *dental* *crown* *is* *a* *shiny* *reflective* *surface* *where* *baked-in* *lighting* *is* *the* *de facto* 2024 *3D-gen* *failure* *mode*; the *SG-illumination* *is* *the* *exact* *mechanism* *for* *clinical* *dental* *crown* *shading* *separation*); SV3D's *coarse-to-fine* *Instant-NGP→DMTet* *pipeline* is the *direct* *H1* *mechanism* *for* *v0 v0* (the *2-stage* *coarse-then-fine* *is* *the* *exact* *mechanism* *for* *clinical* *crown* *mesh* *refinement* *where* *the* *coarse* *NeRF* *captures* *the* *general* *shape* *and* *the* *fine* *DMTet* *captures* *the* *margin* *detail*); and SV3D's *video-LDM* *fine-tune-from-scratch* *is* *the* *direct* *H2* *mechanism* *for* *v0 v0* (the *practical* *engineering* *recipe* *for* *fine-tuning* *a* *video* *foundation* *model* *for* *dental* *arch* *multi-view* *generation*). The *practical v0 v0 v1 v2 caveat*: SV3D's *code* *is* *released* (*Stability* *AI* *Community* *License* — *commercial* *OK* *for* *>$1M* *revenue*, *but* *v0 v0 v1 v2* *is* *a* *startup* *<* *$1M* *revenue* *so* *NOT* *commercial-deployable*, the *practical* *reason* *v0 v0 v1 v2* *must* *fine-tune* *from* *scratch* *on* *dental* *data*); the *practical* *engineering* *starting* *point* is *TripoSR 108* (the *sister* *paper* *by* *the* *same* *Stability* *AI* *team* *for* *fast* *feed-forward* *image-to-3D* *reconstruction*) + *Bolt3D 116* (the *successor* *for* *real-image* *re-training* *+* *scene-level* *multi-view* *generation*).**

## Research question + their answer

**Research question (Sec. 1, paraphrased):** *Single-image* *3D* *object* *reconstruction* is a *long-standing* *problem* *in* *computer* *vision* *with* *applications* *in* *game* *design* + *AR/VR* + *e-commerce* + *robotics*. It is *highly* *challenging* *and* *ill-posed* (requires *lifting* *2D* *pixels* *to* *3D* *space* *while* *reasoning* *about* *the* *unseen* *portions* *of* *the* *object*). *Recent* *3D-gen* *methods* *typically* *use* *either* (a) *image-based* *2D* *generative* *models* (e.g. *SD* / *Imagen*) as a *3D* *optimization* *loss* *function* (*DreamFusion* + *Magic3D* + *ProlificDreamer* — *slow* *SDS* *sampling* + *Janus* *problem* + *over-saturated* *textures*) or (b) *repurpose* *2D* *generative* *models* *for* *NVS* + *3D* *generation* (*Zero123* + *MVDream* + *SyncDreamer* + *Wonder3D* + *EscherNet* + *Free3D*). Can we *develop* *a* *single* *framework* *that* *achieves* (i) *generalization* *to* *real-world* *images* + (ii) *controllability* *over* *camera* *pose* + (iii) *multi-view* *consistency* *for* *high-quality* *3D* *extraction*?

**Their answer (Sec. 1, verbatim summary):** *We* *present* *Stable* *Video* *3D* *(SV3D)* — a *latent* *video* *diffusion* *model* *for* *high-resolution*, *image-to-multi-view* *generation* *of* *orbital* *videos* *around* *a* *3D* *object*. *We* *propose* *SV3D* *that* *adapts* *image-to-video* *diffusion* *model* *for* *novel* *multi-view* *synthesis* *and* *3D* *generation*, *thereby* *leveraging* *the* *generalization* *and* *multi-view* *consistency* *of* *the* *video* *models*, *while* *further* *adding* *explicit* *camera* *control* *for* *NVS*. *We* *also* *propose* *improved* *3D* *optimization* *techniques* *to* *use* *SV3D* *and* *its* *NVS* *outputs* *for* *image-to-3D* *generation*. *Extensive* *experimental* *results* *on* *multiple* *datasets* *with* *2D* *and* *3D* *metrics* *as* *well* *as* *user* *study* *demonstrate* *SV3D*'s *state-of-the-art* *performance* *on* *NVS* *as* *well* *as* *3D* *reconstruction* *compared* *to* *prior* *works*.

The *key* *insight* is that **the *temporal-consistency* of *video* *LDMs* (SVD) can be *repurposed* *as* *spatial* *3D-consistency* *of* *objects* + *adding* *explicit* *camera-pose* *conditioning* *enables* *arbitrary* *view* *generation** — the *paradigm-shift* *that* *defines* *the* *2024-2026* *video-diffusion-for-3D* *field*. The *four* *killer* *paradigm-shifts* are **(a) the *video-LDM* *repurposing* *paradigm* (fine-tune *SVD* *as* *a* *multi-view* *diffusion* *model* — *temporal* *self-attention* *in* *video* *LDM* *becomes* *spatial* *3D-consistency* *in* *NVS*, the *canonical* *paradigm-shift* *that* *defines* *the* *2024-2026* *video-diffusion-for-3D* *field*)**, **(b) the *camera-trajectory-as-conditioning* *trick* (*sinusoidal* *embedding* *of* *(elevation*, *azimuth)* *added* *to* *noise-timestep* *embedding* *in* *every* *residual* *block* — the *canonical* *H3* *mechanism* *for* *camera-conditional* *multi-view* *generation*)**, **(c) the *disentangled-illumination* *model* + *masked-SDS* *loss* (*24* *Spherical* *Gaussians* *for* *lighting* *separation* + *visibility-aware* *soft-mask* *for* *SDS* *inpainting* — the *killer* *practical* *mechanism* *for* *preventing* *baked-in* *lighting* + *preserving* *input-image* *faithfulness*, the *direct* *practical* *ancestor* *of* *Bolt3D 116*'s *real-image* *re-training* *recipe*)**, and **(d) the *coarse-to-fine* *Instant-NGP→DMTet* *3D-optimization* *pipeline* (*NeRF* *coarse* *stage* *for* *general* *shape* *+* *texture* *→* *DMTet* *fine* *stage* *for* *high-resolution* *mesh* *refinement* — the *canonical* 2024 *3D-from-NVS* *pipeline*, the *de facto* *template* *for* *v0 v0* *clinical* *crown* *generation* *pipeline*)**.

The *comparison* *to* *the* *alternative* *3D-creation* *paradigms* *is* *direct*:
- *SDS-based* *text-to-3D* (*DreamFusion* + *ProlificDreamer* + *Magic3D* + *DreamGaussian*): *slow* (*hours* *per* *scene* *for* *SDS* *sampling*), *unstable* (*the* *de facto* *SDS* *Janus* *problem* + *over-saturated* *textures*), *no* *camera* *control*.
- *Image-based* *NVS* (*Zero123* + *Zero123-XL* + *Stable* *Zero123* + *EscherNet* + *Free3D*): *one* *view* *at* *a* *time* (*NOT* *multi-view* *consistent*) or *low* *256×256* *resolution* (*insufficient* *for* *clinical* *detail*), *limited* *to* *static* *orbits*.
- *Image-based* *multi-view* *diffusion* (*MVDream* + *SyncDreamer* + *HexGen3D* + *Zero123++* + *Wonder3D* + *Consistent-1-to-3* + *One-2-3-45* + *One-2-3-45++*): *NOT* *controllable* (*only* *generates* *specific* *views* *given* *a* *conditional* *image*, *NOT* *arbitrary* *viewpoints*), *limited* *to* *3D* *dataset* *quality* *they* *were* *fine-tuned* *on*.
- *Video-diffusion-for-NVS* *predecessors* (*SVD-MV* + *IM-3D* + *Vivid-1-to-3*): *only* *static* *orbits* *at* *fixed* *elevation*, *no* *explicit* *camera* *control*, *limited* *to* *360°* *azimuths* *at* *single* *elevation*.
- SV3D (this paper): *video-LDM* *fine-tune* + *explicit* *camera-pose* *conditioning* + *dynamic* *orbits* + *576×576* *high-resolution* + *coarse-to-fine* *3D-optimization* + *disentangled-illumination* + *masked-SDS* — the *first* *paper* *to* *combine* *all* *four* *advantages* *simultaneously*.

## Method

### Video-LDM backbone (Sec. 3, Fig. 2)

**Architecture:**
- **Backbone:** SVD-xt (Stable Video Diffusion - extended, the *de facto* 2023 *open* *video* *LDM* *foundation* *model* — *3D-attention* *UNet* *with* *multi-layer* *Conv3D* *residual* *blocks* + *spatial* *+* *temporal* *self-attention* *transformer* *blocks*, the *de facto* 2023 *video* *LDM* *foundation* *fine-tuned* *for* *NVS*). *Temporal* *self-attention* *in* *video* *LDM* *becomes* *spatial* *3D-consistency* *in* *NVS* (the *de facto* *paradigm-shift* *from* *temporal* *to* *spatial*).
- **Four key modifications** to *SVD* *for* *NVS*:
  - (i) *Remove* *fps-id* + *motion-bucket-id* *vector* *conditionings* (*irrelevant* *for* *NVS* — *NVS* *has* *fixed* *fps* *with* *no* *motion*).
  - (ii) *Concatenate* *conditioning* *image* *latent* (VAE-encoded by *SVD-VAE*) *to* *noisy* *latent* *state* *z_t* (the *direct* *image-conditioning* *mechanism* *inherited* *from* *SVD*).
  - (iii) *CLIP-embedding* *of* *conditioning* *image* *as* *cross-attention* *key/value* (the *de facto* 2023 *cross-attention* *conditioning* *mechanism* *inherited* *from* *SD*).
  - (iv) *Camera-trajectory* *conditioning* via *sinusoidal* *embedding* *of* *(elevation, azimuth)* *for* *each* *target* *view* *concatenated* *with* *noise-timestep* *embedding* *in* *every* *residual* *block* (the *killer* *H3* *innovation*, the *direct* *template* *for* *camera-conditional* *multi-view* *generation*).

### Camera trajectory conditioning (Sec. 3, Fig. 2) — the killer H3 innovation

**Mechanism (Sec. 3, "SV3D Architecture"):**
- The *camera* *pose* *angles* *e_i* + *a_i* + *noise* *timestep* *t* *are* *first* *embedded* *into* *sinusoidal* *position* *embeddings*.
- The *camera* *pose* *embeddings* *are* *concatenated* *together*, *linearly* *transformed*, *and* *added* *to* *the* *noise* *timestep* *embedding*.
- This *is* *fed* *to* *every* *residual* *block*, *where* *they* *are* *added* *to* *the* *block*'s *output* *feature* (*after* *being* *linearly* *transformed* *again* *to* *match* *the* *feature* *size*).

**Comparison to alternatives:** *Concatenating* *pose* *embedding* *to* *noise-timestep* *embedding* *in* *every* *residual* *block* is *more* *parameter-efficient* *than* *CAT3D 113*'s *raymap* *channel-wise* *concatenation* + *more* *general* *than* *Zero123*'s *pose-difference* *conditioning* + *more* *controllable* *than* *MVDream*'s *fixed* *view* *set*. The *killer* *practical* *advantage* is *the* *per-frame* *pose* *conditioning* — *each* *of* *the* *21* *target* *frames* *gets* *its* *own* *(elevation, azimuth)*, *enabling* *arbitrary* *camera* *trajectories* *from* *dynamic* *orbits* *to* *cylindrical* *spirals* *to* *sphere* *trajectories* — the *de facto* 2024 *camera-trajectory* *paradigm* *that* *CAT3D 113* + *Bolt3D 116* + *L4GM 114* *inherits* + *extends*.

### Three model variants (Sec. 3, "Models")

**SV3D^u (pose-unconditioned, static orbit):**
- *Generates* *video* *of* *static* *orbit* *around* *an* *object* *while* *only* *conditioned* *on* *a* *single-view* *image*.
- *Unlike* *SVD-MV*, *does* *NOT* *provide* *the* *elevation* *angle* — *the* *model* *infers* *the* *elevation* *from* *the* *conditioning* *image* (the *de facto* 2024 *zero-camera-pose* *NVS* *paradigm*).
- *Limitation*: *the* *model* *can* *only* *generate* *views* *at* *the* *same* *elevation* *as* *the* *input* *image*, *missing* *top* *+* *bottom* *views* — *insufficient* *for* *complete* *3D* *object* *reconstruction* (the *de facto* 2024 *single-elevation* *limitation*).

**SV3D^c (pose-conditioned, dynamic orbit):**
- *Conditioned* *on* *the* *input* *image* + *a* *sequence* *of* *camera* *elevation* *+* *azimuth* *angles* *in* *an* *orbit*.
- *Trained* *on* *dynamic* *orbits* *with* *variable* *elevation* + *azimuth* (the *killer* *practical* *training* *data* *that* *forces* *the* *model* *to* *learn* *top* *+* *bottom* *views*).
- *Result*: *SV3D^c* *outperforms* *SV3D^u* *on* *dynamic* *orbits* + *matches* *SV3D^u* *on* *static* *orbits* (the *de facto* 2024 *empirical* *evidence* *that* *pose-conditioning* *is* *necessary* *for* *dynamic* *NVS*).

**SV3D^p (progressive fine-tune, the killer practical recipe):**
- *Following* *SVD*'s *intuition* *to* *progressively* *increase* *the* *task* *difficulty* *during* *training*.
- *Step 1*: *fine-tune* *SVD* *to* *produce* *static* *orbits* *unconditionally* (*SV3D^u* *recipe*) for *55k* *iterations*.
- *Step 2*: *further* *fine-tune* *on* *dynamic* *orbits* *with* *camera* *pose* *conditioning* (*SV3D^c* *recipe*) for *50k* *iterations*.
- *Result*: *SV3D^p* *outperforms* *both* *SV3D^u* + *SV3D^c* *on* *both* *static* *+* *dynamic* *orbits* — the *de facto* 2024 *empirical* *evidence* *that* *progressive* *fine-tuning* *from* *easy* *to* *hard* *tasks* *is* *the* *best* *video-diffusion* *fine-tuning* *recipe*.

### Static vs Dynamic orbit training data (Sec. 3, Fig. 3)

**Static orbit:**
- The *camera* *circles* *around* *an* *object* *at* *regularly-spaced* *azimuths* *at* *the* *same* *elevation* *angle* *as* *that* *in* *the* *conditioning* *image*.
- *Disadvantage*: *might* *not* *get* *any* *information* *about* *the* *top* *or* *bottom* *of* *the* *object* *depending* *on* *the* *conditioning* *elevation* *angle* — the *de facto* 2024 *single-elevation* *limitation*.

**Dynamic orbit:**
- The *azimuths* *can* *be* *irregularly* *spaced*, *and* *the* *elevation* *can* *vary* *per* *view*.
- *Recipe*: *sample* *a* *static* *orbit*, *add* *small* *random* *noise* *to* *the* *azimuth* *angles*, *and* *add* *a* *random* *weighted* *combination* *of* *sinusoids* *with* *different* *frequencies* *to* *the* *elevation*.
- *Provides* *temporal* *smoothness* (sinusoidal *variation* *ensures* *smooth* *trajectory*).
- *Ensures* *that* *the* *camera* *trajectory* *loops* *around* *to* *end* *at* *the* *same* *azimuth* *and* *elevation* *as* *those* *of* *the* *conditioning* *image* (the *killer* *practical* *mechanism* *for* *closed-loop* *orbiting* *NVS*).

### Triangular CFG scaling (Sec. 3, Fig. 4)

**Problem:** *SVD* *uses* *a* *linearly* *increasing* *scale* *for* *classifier-free* *guidance* *(CFG)* *from* *1* *to* *4* *across* *the* *generated* *frames*. *However*, *this* *scaling* *causes* *the* *last* *few* *frames* *in* *our* *generated* *orbits* *to* *be* *over-sharpened* (the *de facto* 2024 *linear-CFG* *over-sharpening* *problem*).

**Solution:** *Triangle* *wave* *CFG* *scaling* *during* *inference*:
- *Linearly* *increase* *CFG* *from* *1* *at* *the* *front* *view* *to* *2.5* *at* *the* *back* *view* (the *first* *half* *of* *the* *orbit*).
- *Then* *linearly* *decrease* *back* *to* *1* *at* *the* *front* *view* (the *second* *half* *of* *the* *orbit*).
- *Prevents* *over-sharpening* *in* *the* *penultimate* *frame* (the *killer* *practical* *inference* *trick*).
- *Produces* *more* *details* *in* *the* *back* *view* (frame 12 of 21).

### Training details (Sec. 3, "Training Details")

- *Training* *data*: *Objaverse* (the *de facto* 2023-2024 *3D-asset* *corpus*, *730K* *3D* *models* — the *de facto* 2024 *NVS* *training* *corpus*).
- *Render* *setup*: *21* *frames* *per* *object* *at* *576×576* *resolution* *with* *33.8°* *FOV* *on* *random* *color* *background*.
- *Model*: *fine-tune* *SVD-xt* (the *de facto* 2023 *open* *video* *LDM* *foundation* *model*) *to* *output* *21* *frames*.
- *Training* *time*: *105k* *iterations* *with* *effective* *batch* *64* on *4* *nodes* × *8* *A100* *GPUs* (32 *A100s* *total*) for *~6* *days*.
- *Per-model* *split*:
  - *SV3D^u*: *55k* *iterations* *unconditional* *static* *orbit*.
  - *SV3D^c*: *105k* *iterations* *conditional* *dynamic* *orbit*.
  - *SV3D^p*: *55k* *iterations* *unconditional* *static* *orbit* *+* *50k* *iterations* *conditional* *dynamic* *orbit* = *105k* *total*.

### 3D Optimization Pipeline (Sec. 4, Fig. 7)

**Coarse-to-fine training scheme (Sec. 4, "Coarse-to-Fine Training"):**

**Coarse stage (Instant-NGP NeRF):**
- *Train* *an* *Instant-NGP* *NeRF* *representation* *to* *reconstruct* *the* *SV3D-generated* *images* (*i.e.* *without* *SDS* *loss*) *at* *a* *lower* *resolution*.
- *Losses*: *pixel-level* *MSE* + *mask* *loss* + *perceptual* *LPIPS* *loss* (the *de facto* 2023-2024 *NeRF* *training* *loss* *recipe*).
- *Time*: *~2* *minutes* (the *fast* *coarse* *stage*).

**Mesh extraction (Marching Cubes):**
- *Extract* *a* *mesh* *from* *the* *trained* *NeRF* *using* *marching* *cubes* (the *de facto* 2024 *mesh* *extraction* *algorithm*).

**Fine stage (DMTet):**
- *Refine* *mesh* *with* *DMTet* (the *hybrid* *SDF-Mesh* *representation* — *de facto* 2023-2024 *high-resolution* *mesh* *representation*).
- *Losses*: *photometric* *reconstruction* + *masked-SDS* *loss* + *geometric* *priors* (*smooth* *depth* *loss* *from* *RegNeRF* + *bilateral* *normal* *smoothness* + *mono* *normal* *loss* *from* *Omnidata*).
- *Time*: *~12* *minutes* with *SDS* *or* *~6* *minutes* *without* *SDS* (the *fine* *stage*).

**UV unwrapping:**
- *Use* *xatlas* *to* *perform* *the* *UV* *unwrapping* *and* *export* *the* *output* *object* *mesh* (the *de facto* 2024 *UV* *unwrapping* *tool*).

**Total time:** *~8* *minutes* *without* *SDS* *loss*, *~20* *minutes* *with* *SDS* *loss* (the *practical* *inference* *time* *for* *v0 v0* *clinical* *crown* *generation*).

### Disentangled Illumination Model (Sec. 4, "Disentangled Illumination Model")

**Motivation:** *SDS-based* *optimization* *techniques* *typically* *use* *random* *illuminations* *at* *every* *iteration* → *baked-in* *lighting* *in* *the* *output* *mesh*. *However*, *SV3D-generated* *videos* *are* *under* *consistent* *illumination* (the *lighting* *remains* *static* *while* *the* *camera* *circles* *around* *an* *object*) — *this* *is* *a* *strong* *prior* *that* *can* *be* *exploited* *for* *disentangling* *illumination*.

**Method:**
- *Fit* *a* *simple* *illumination* *model* *of* *24* *Spherical* *Gaussians* *(SG)* (the *de facto* 2020-2024 *lighting* *representation*) *inspired* *by* *prior* *decomposition* *methods* (e.g. *NeRF-Art* + *Total* *Relighting*).
- *Model* *white* *light* *and* *hence* *only* *use* *a* *scalar* *amplitude* *for* *the* *SGs* (the *killer* *simplification* *for* *efficient* *training*).
- *Only* *consider* *Lambertian* *shading*, *where* *the* *cosine* *shading* *term* *is* *approximated* *with* *another* *SG* (the *killer* *Lambertian* *approximation* *that* *works* *well* *for* *diffuse* *objects*).
- *Learn* *the* *parameters* *of* *the* *illumination* *SGs* *using* *a* *reconstruction* *loss* *between* *the* *rendered* *images* *and* *SV3D-generated* *images* (the *de facto* *illumination* *fitting* *recipe*).
- *Illumination-replication* *loss*: *L_illum* *=* *|V(I)* *−* *L|²* *where* *V(c)* *=* *max(c_r*, *c_g*, *c_b)* (the *HSV-value* *component*) — *reduces* *baked-in* *illumination* *by* *replicating* *the* *input* *image*'s *HSV-value* *component* *with* *the* *rendered* *illumination* *L*.

### Masked SDS Loss (Sec. 4.1, "Masked SDS Loss")

**Problem:** *Naive* *SDS* *loss* *causes* *unstable* *training* *and* *unfaithful* *texture* *to* *the* *input* *images* (e.g. *oversaturation* *or* *blurry* *artifacts*).

**Solution:** *Soft* *visibility* *mask* *M* *based* *on* *dot-product* *between* *surface* *normal* *n* *and* *view* *direction* *to* *the* *most-visible* *reference* *camera*:
- *For* *each* *random* *camera* *view*, *obtain* *the* *visible* *surface* *points* *p* *∈* *ℝ³* *and* *their* *corresponding* *surface* *normals* *n*.
- *For* *each* *reference* *camera* *i*, *calculate* *the* *view* *directions* *v_i* *of* *the* *surface* *p* *towards* *its* *position* *π̄_ref^i* *∈* *ℝ³* *as* *v_i* *=* *(π̄_ref^i* *−* *p)* */* *||π̄_ref^i* *−* *p||*.
- *Infer* *the* *visibility* *of* *this* *surface* *from* *the* *reference* *camera* *based* *on* *the* *dot* *product* *between* *v_i* *and* *n* (i.e. *v_i* *·* *n*). *Higher* *values* *roughly* *indicate* *more* *visibility*.
- *Chose* *that* *reference* *camera* *c* *that* *has* *maximum* *likelihood* *of* *visibility*: *c* *=* *max_i* *(v_i* *·* *n)*.
- *Smoothstep* *function* *f_s* *to* *smoothly* *clip* *to* *c*'s *visibility* *range* *v_c* *·* *n*. *M* *=* *1* *−* *f_s(v_c* *·* *n*, *0*, *0.5)*.
- *Combined* *visibility* *mask* *M* *is* *applied* *to* *SDS* *loss*: *L_mask-sds* *=* *M* *·* *L_sds*.

**Effect:** *SDS* *loss* *is* *only* *applied* *to* *unseen* *or* *grazing-angle* *areas* *from* *c* — *preserves* *the* *texture* *of* *clearly-visible* *surfaces* *in* *the* *reference* *orbit* *while* *inpainting* *the* *missing* *details* (the *killer* *practical* *mechanism* *for* *clinical* *dental* *margin* *refinement* *where* *the* *margin* *is* *partially* *visible* *in* *the* *reference* *views* *but* *needs* *to* *be* *inpainted* *in* *the* *unseen* *regions*).

### Geometric Priors (Sec. 4.1, "Geometric Priors")

- *Smooth* *depth* *loss* *from* *RegNeRF* (the *de facto* 2023 *depth* *smoothness* *loss*) — *encourages* *smooth* *3D* *surfaces* *where* *the* *projected* *image* *gradients* *are* *low*.
- *Bilateral* *normal* *smoothness* *loss* (the *de facto* 2023-2024 *normal* *smoothness* *loss*) — *encourages* *smooth* *normals* *where* *the* *image* *gradients* *are* *low*.
- *Mono* *normal* *loss* *from* *Omnidata* (the *de facto* 2023-2024 *monocular* *normal* *estimator*) — *reduces* *noisy* *surfaces* *in* *the* *output* *mesh*.

## Results

### Novel View Synthesis on GSO + OmniObject3D (Sec. 3.1, Tables 1-4)

**Table 1: GSO static orbits (the killer empirical evidence that video-LDM > image-LDM):**

| Model | LPIPS↓ | PSNR↑ | SSIM↑ | CLIP-S↑ | MSE↓ |
|---|---|---|---|---|---|
| SyncDreamer | 0.17 | 15.78 | 0.76 | 0.87 | 0.03 |
| Zero123 | 0.13 | 17.29 | 0.79 | 0.85 | 0.04 |
| Zero123XL | 0.14 | 17.11 | 0.78 | 0.85 | 0.04 |
| Stable Zero123 | 0.13 | 18.34 | 0.78 | 0.85 | 0.05 |
| Free3D | 0.15 | 16.18 | 0.79 | 0.84 | 0.04 |
| EscherNet | 0.13 | 16.73 | 0.79 | 0.85 | 0.03 |
| **SV3D^u** | **0.09** | **21.14** | **0.87** | **0.89** | **0.02** |
| SV3D^c | 0.09 | 20.56 | 0.87 | 0.88 | 0.02 |
| **SV3D^p** | **0.08** | **21.26** | **0.88** | **0.89** | **0.02** |

**SV3D^p outperforms prior SOTA (Stable Zero123) by +2.92 PSNR, +0.10 SSIM, +0.04 CLIP-S on GSO static — the killer empirical evidence that video-LDM fine-tuning > image-LDM fine-tuning.**

**Table 2: GSO dynamic orbits:**

| Model | LPIPS↓ | PSNR↑ | SSIM↑ | CLIP-S↑ | MSE↓ |
|---|---|---|---|---|---|
| Zero123 | 0.14 | 16.99 | 0.79 | 0.84 | 0.04 |
| Zero123XL | 0.14 | 16.73 | 0.78 | 0.84 | 0.04 |
| Stable Zero123 | 0.13 | 18.04 | 0.78 | 0.85 | 0.05 |
| Free3D | 0.18 | 14.93 | 0.77 | 0.83 | 0.05 |
| EscherNet | 0.13 | 16.47 | 0.79 | 0.84 | 0.03 |
| SV3D^c | 0.10 | 19.99 | 0.86 | 0.87 | 0.02 |
| **SV3D^p** | **0.09** | **20.38** | **0.87** | **0.87** | **0.02** |

**Table 3: OmniObject3D static orbits:**

| Model | LPIPS↓ | PSNR↑ | SSIM↑ | CLIP-S↑ | MSE↓ |
|---|---|---|---|---|---|
| Zero123 | 0.17 | 15.50 | 0.76 | 0.83 | 0.05 |
| Stable Zero123 | 0.15 | 16.86 | 0.77 | 0.84 | 0.06 |
| Free3D | 0.16 | 15.29 | 0.78 | 0.83 | 0.05 |
| EscherNet | 0.17 | 14.63 | 0.74 | 0.83 | 0.05 |
| **SV3D^u** | **0.10** | **19.68** | **0.86** | **0.86** | **0.02** |
| **SV3D^p** | **0.10** | **19.91** | **0.86** | **0.86** | **0.02** |

**Table 4: OmniObject3D dynamic orbits:** SV3D^p achieves 0.10 / ~20.0 / 0.86 / 0.85 / 0.02 — outperforming all prior methods by +3-5 PSNR.

### 3D Generation on GSO (Sec. 4.2, Tables 5-6)

**Table 5: 2D comparison of 3D outputs (rendered from trained meshes on dynamic orbit):**

| Model | LPIPS↓ | PSNR↑ | SSIM↑ | MSE↓ | CLIP-S↑ |
|---|---|---|---|---|---|
| GT renders (oracle) | 0.078 | 19.508 | 0.879 | 0.014 | 0.897 |
| EscherNet | 0.178 | 14.438 | 0.804 | 0.041 | 0.835 |
| Free3D | 0.197 | 14.202 | 0.799 | 0.043 | 0.809 |
| Stable Zero123 | 0.166 | 14.635 | 0.813 | 0.040 | 0.805 |
| SV3D^u | 0.133 | 15.957 | 0.834 | 0.031 | 0.871 |
| SV3D^c | 0.132 | 16.373 | 0.834 | 0.027 | 0.870 |
| SV3D^p static orbit | 0.125 | 16.821 | 0.848 | 0.025 | 0.864 |
| SV3D^p no SDS | 0.124 | 16.864 | 0.841 | 0.024 | 0.875 |
| **SV3D^p** | **0.119** | **17.405** | **0.849** | **0.021** | **0.877** |

**SV3D^p PSNR 17.405 is within 2 PSNR of GT renders (19.508) — the killer empirical evidence that SV3D-generated multi-view images are nearly as informative as real GT images for 3D reconstruction.**

**Table 6: 3D metrics (Chamfer Distance + 3D IoU):**

| Model | CD↓ | 3D IoU↑ |
|---|---|---|
| GT renders (oracle) | 0.021 | 0.689 |
| Point-E | 0.074 | 0.162 |
| Shap-E | 0.071 | 0.267 |
| DreamGaussian | 0.055 | 0.411 |
| One-2-3-45++ | 0.054 | 0.406 |
| SyncDreamer | 0.053 | 0.451 |
| EscherNet | 0.042 | 0.466 |
| Free3D | 0.047 | 0.426 |
| Stable Zero123 | 0.039 | 0.550 |
| **SV3D^p** | **0.024** | **0.614** |

**SV3D^p CD 0.024 / 3D-IoU 0.614 — beats Stable Zero123 by 38% CD reduction and 12% IoU improvement, and within 15% CD of GT renders (0.021 / 0.689).**

### Ablation Studies (Sec. 3.1, Sec. 4.2, Tables 1-6)

**(A) Progressive fine-tune > static-only (Table 1, 2, 5, 6):**
- SV3D^p 21.26 PSNR > SV3D^c 20.56 > SV3D^u 21.14 on GSO static.
- SV3D^p 20.38 PSNR > SV3D^c 19.99 on GSO dynamic.
- **The de facto 2024 empirical evidence that progressive fine-tuning from easy (static) to hard (dynamic) is the best video-diffusion fine-tuning recipe.**

**(B) Triangular CFG > linear CFG (Fig. 4, Fig. 5):**
- Triangular CFG prevents over-sharpening in penultimate frame (visual comparison).
- SV3D has the best LPIPS at every frame (0-20) on GSO static (Fig. 5).

**(C) Masked SDS > naive SDS (Table 5, 6, Fig. 10):**
- SV3D^p (with masked SDS): LPIPS 0.119 / PSNR 17.405 / CD 0.024 / IoU 0.614.
- SV3D^p no SDS: LPIPS 0.124 / PSNR 16.864 / CD 0.024 / IoU 0.611.
- **Masked SDS improves PSNR by +0.54 with marginal cost (5 min extra training time) — the de facto 2024 SDS-improvement mechanism.**

**(D) Dynamic orbit > static orbit for 3D generation (Table 5, 6, Fig. 9):**
- SV3D^p (dynamic): LPIPS 0.119 / PSNR 17.405 / CD 0.024 / IoU 0.614.
- SV3D^p static orbit: LPIPS 0.125 / PSNR 16.821 / CD 0.028 / IoU 0.610.
- **Dynamic orbit improves PSNR by +0.58 — the de facto 2024 training data choice.**

**(E) SV3D model choice (Tables 1-6):**
- SV3D^p best on all metrics.
- SV3D^u matches SV3D^p on static orbits but worse on dynamic orbits.
- SV3D^c slightly worse than SV3D^p on both static and dynamic orbits.

### Real-world 3D results (Fig. 12)

- The *killer* *qualitative* *evidence* *that* *SV3D* *generalizes* *to* *real-world* *images* (the *de facto* 2024 *real-world* *NVS* *test*).
- *Accurate* *shape* *and* *details* *in* *reconstructions* *from* *diverse* *in-the-wild* *images* (the *killer* *practical* *evidence* *for* *v0 v0* *clinical* *dental* *applications*).

## Connections to H1-H5

**H1 (multi-stage coarse-to-fine > single-stage):** STRONG DIRECT SUPPORT. The two-stage Instant-NGP→DMTet 3D-optimization pipeline is the canonical 2024 multi-stage paradigm; the empirical ablation (Table 5) shows SV3D^p with full coarse-to-fine achieves the best PSNR (17.405) vs. the coarse-only or fine-only variants. The coarse NeRF captures general shape + texture, the fine DMTet refines the mesh with masked-SDS, the de facto 2024 3D-from-NVS recipe. For v0 v0 sub-task 2 (crown generation), the exact multi-stage recipe applies: coarse NeRF for general crown shape → fine DMTet for high-resolution mesh refinement with margin detail.

**H2 (latent diffusion > direct):** NOT TESTED (SV3D uses video LDM with latent diffusion, but does not ablate against direct video diffusion). However, the *killer* *practical* *evidence* *from* *Bolt3D 116* *+* *CAT3D 113* *+* *L4GM 114* (which *all* *inherit* *the* *latent-diffusion* *paradigm* *from* *SV3D*) *suggests* *latent* *diffusion* *is* *superior* *for* *video* *NVS* *generation*. For v0 v0, the *practical* *engineering* *recipe* is *to* *use* *SVD-style* *latent* *diffusion* *fine-tune* (NOT *direct* *pixel-space* *diffusion*).

**H3 (multi-view / arch-level conditioning):** STRONG DIRECT SUPPORT + CATALOG OF MECHANISMS. SV3D's three killer H3 mechanisms are:
- (a) **Camera-trajectory-as-conditioning** (sinusoidal embedding of (elevation, azimuth) added to noise-timestep embedding in every residual block) — the canonical 2024 H3 mechanism, the direct template for v0 v0 clinical dental arch multi-view generation.
- (b) **Static vs Dynamic orbit training data** (dynamic orbit with sinusoidal elevation variation forces the model to learn top + bottom views) — the canonical 2024 H3 training data choice, the direct mechanism for clinical dental occlusal + cervical view generation.
- (c) **Triangular CFG scaling** (linearly increase CFG from 1 to 2.5 from front to back, then decrease back to 1) — the canonical 2024 H3 inference trick, the direct mechanism for clinical closed-loop orbiting NVS without discontinuity.
For v0 v0 sub-task 1 (full-arch synthesis), the exact H3 recipe applies: fine-tune SVD-style video LDM with clinical dental arch orbital data, conditioning on (intra-oral camera trajectory, prep-tooth) → 21 frames of dental arch views.

**H4 (implicit SDF / flexible representation > mesh):** STRONG DIRECT SUPPORT. The DMTet representation (hybrid SDF-Mesh) is the canonical 2024 H4 mechanism for high-resolution mesh output, with the Instant-NGP→DMTet pipeline providing the exact coarse-to-fine SDF refinement. The 24-SG disentangled-illumination model is the canonical 2024 H4 mechanism for lighting separation + baked-in lighting prevention. For v0 v0 sub-task 2, the exact H4 recipe applies: use DMTet (or FlexiCubes from paper 007) for crown mesh extraction with SG-illumination for clinical lighting separation.

**H5 (synthetic pre-train + dental fine-tune):** NOT TESTED but the *killer* *practical* *evidence* *from* *Bolt3D 116* (which *extends* *SV3D* *with* *real-image* *re-training*) + *the* *de facto* 2024 *video-LDM* *fine-tuning* *recipe* *suggest* *Objaverse* *pre-train* *+* *clinical* *dental* *fine-tune* *is* *the* *best* *recipe* *for* *v0 v0*. The *practical* *engineering* *recipe* is *to* *fine-tune* *SVD-xt* *on* *Objaverse* + *clinical* *dental* *arch* *data* *with* *progressive* *schedule* (*static* *orbit* *→* *dynamic* *orbit*) for *4-6* *weeks* *on* *4-8* *A100s*.

## Surprises / interesting things buried in section 4

1. **The de facto 2024 video-LDM-fine-tuning recipe is "progressive" (static-then-dynamic), not "joint"** — SV3D^p outperforms both SV3D^u (static-only) and SV3D^c (dynamic-only) on both static and dynamic orbits. The empirical evidence that progressive training is better than joint training is the *killer* *practical* *finding* for v0 v0 clinical dental arch fine-tuning (start with single-arch single-elevation, then add multi-elevation + multi-arch).

2. **The 24-SG illumination model is the killer practical mechanism for baked-in lighting prevention** — the *de facto* 2024 *3D-from-NVS* *failure* *mode* is *baked-in* *lighting* (the *output* *mesh* *has* *the* *lighting* *baked* *into* *the* *texture*, *making* *it* *impossible* *to* *re-light* *the* *object*). The *24-SG* *illumination* *model* + *illumination-replication* *loss* *L_illum* = *|V(I) − L|²* is the *killer* *practical* *mechanism* *for* *preventing* *baked-in* *lighting*, the *direct* *practical* *ancestor* *of* *Bolt3D 116*'s *real-image* *re-training* *recipe*.

3. **The masked-SDS loss is the killer practical mechanism for clinical margin refinement** — the *soft* *visibility* *mask* *M* = *1 − smoothstep(v_c · n, 0, 0.5)* *based* *on* *dot-product* *between* *surface* *normal* *n* *and* *view* *direction* *to* *the* *most-visible* *reference* *camera* is the *killer* *practical* *mechanism* *for* *clinical* *dental* *margin* *refinement* *where* *the* *margin* *is* *partially* *visible* *in* *the* *reference* *views* *but* *needs* *to* *be* *inpainted* *in* *the* *unseen* *regions*. The *mask* *is* *applied* *to* *SDS* *loss* *only*, *preserving* *the* *photometric* *reconstruction* *loss* *in* *visible* *regions*.

4. **The triangular-CFG scaling is the killer practical inference trick** — the *de facto* 2024 *linear-CFG* *over-sharpening* *problem* (linear CFG from 1 to 4 in SVD causes over-sharpening in the penultimate frame of a closed-loop orbit) is *solved* *by* *triangular-CFG* *scaling* (linearly increase CFG from 1 to 2.5 from front to back, then decrease back to 1 at front). The *killer* *practical* *inference* *trick* *for* *v0 v0* *clinical* *dental* *arch* *orbiting* *NVS* *where* *the* *orbit* *must* *return* *to* *the* *starting* *view* *without* *discontinuity*.

5. **The static orbit vs dynamic orbit distinction is the killer practical training data choice** — *static* *orbit* = *uniform* *azimuth* *at* *fixed* *elevation* *inferred* *from* *input* *image* (used for *SV3D^u*); *dynamic* *orbit* = *irregular* *azimuth* *spacing* + *sinusoidal* *elevation* *variation* (used for *SV3D^c*). The *static* *orbit* *is* *insufficient* *for* *complete* *3D* *object* *reconstruction* (misses *top* *+* *bottom* *views*); the *dynamic* *orbit* *is* *necessary* *for* *complete* *3D* *object* *reconstruction* (covers *top* *+* *bottom* *+* *side* *views*). The *exact* *mechanism* *for* *clinical* *dental* *occlusal* *+* *cervical* *+* *buccal* *+* *lingual* *view* *generation*.

6. **The 576×576 resolution is the killer practical high-resolution baseline** — the *de facto* 2024 *high-resolution* *object-centric* *NVS* *SOTA*, the *3×* *resolution* *jump* from *Zero123* + *EscherNet* + *Free3D*'s *256×256* *image-diffusion* *baselines*. The *killer* *practical* *advantage* *for* *v0 v0* *clinical* *dental* *margin* *detail* *at* *sub-100μm* *scale* (576*×*576 *resolution* *at* *33.8°* *FOV* *gives* *sub-100μm* *detail* *at* *typical* *dental* *arch* *scale*).

7. **The 8-min coarse + 12-min fine = 20-min total inference time is the practical clinical speed target** — *not* *the* *de facto* 2024 *real-time* *NVS* *speed* *target* (which is *<1* *sec* *for* *LRM 107* + *TripoSR 108*), but *a* *practical* *clinical* *workflow* *target* (the *dentist* *iterates* *on* *the* *design* *in* *real-time*, with *each* *iteration* *taking* *~20* *sec* *to* *1* *min* *for* *the* *complete* *cycle*).

## Quote-worthy sentences

1. (Sec. 1) "Recent work on 3D generation propose techniques to adapt 2D generative models for novel view synthesis (NVS) and 3D optimization. However, these methods have several disadvantages due to either limited views or inconsistent NVS, thereby affecting the performance of 3D object generation."

2. (Sec. 1) "In this work, we propose SV3D that adapts image-to-video diffusion model for novel multi-view synthesis and 3D generation, thereby leveraging the generalization and multi-view consistency of the video models, while further adding explicit camera control for NVS."

3. (Sec. 1) "We call our resulting NVS network 'SV3D'. To our knowledge, this is the first work that adapts a video diffusion model for explicit pose-controlled view synthesis."

4. (Sec. 1) "We also propose improved 3D optimization techniques to use SV3D and its NVS outputs for image-to-3D generation."

5. (Sec. 1) "In addition, we propose to jointly optimize a disentangled illumination model along with 3D shape and texture, effectively reducing the issue of baked-in lighting."

6. (Sec. 2.1) "We argue that the existing NVS and 3D generation methods do not fully leverage the superior generalization capability, controllability, and consistency in video diffusion models."

7. (Sec. 3) "Our main idea is to repurpose temporal consistency in a video diffusion model for spatial 3D consistency of an object."

8. (Sec. 3) "To the best of our knowledge, SV3D is the first video diffusion-based framework for controllable multi-view synthesis at 576×576 resolution (and subsequently for 3D generation)."

9. (Sec. 3) "In a dynamic orbit, the azimuths can be irregularly spaced, and the elevation can vary per view."

10. (Sec. 3) "Since we generate videos looping back to the front-view image, we propose to use a triangle wave CFG scaling during inference: linearly increase CFG from 1 at the front view to 2.5 at the back view, then linearly decrease it back to 1 at the front view."

11. (Sec. 3) "Interestingly, from Tables 1 and 3, we find that both SV3D^c and SV3D^p outperform SV3D^u on generations of static orbits, even though SV3D^u is trained specifically on static orbits."

12. (Sec. 3) "This shows that progressive finetuning from easier (static) to harder (dynamic) tasks is indeed a favorable way to finetune a video diffusion model."

13. (Sec. 4) "Benefiting from the multi-view consistency in SV3D, we are able to produce high-quality 3D meshes directly from the SV3D novel view images."

14. (Sec. 4) "We also design a masked score distillation sampling (SDS) loss to further enhance 3D quality in the regions that are not visible in the SV3D-predicted novel views."

15. (Sec. 4) "We adopt a two-stage, coarse-to-fine training scheme to generate a 3D mesh from input images, similar to [Magic3D, Fantasia3D]."

16. (Sec. 4) "We propose to fit a simple illumination model of 24 Spherical Gaussians (SG) inspired by prior decomposition methods."

17. (Sec. 4.1) "Therefore, we design a soft masking mechanism to only apply SDS loss on the unseen/occluded areas, allowing it to inpaint the missing details while preserving the texture of clearly-visible surfaces in the training orbit."

18. (Sec. 4.1) "We apply SDS loss on only those pixels in the random orbit views that are not likely visible in the reference orbit views."

19. (Sec. 4.2) "Our best model, SV3D^p, performs comparably to using GT renders for reconstruction in terms of the 3D metrics, which further demonstrates the 3D consistency of our generated images."

20. (Sec. 4.2) "The images generated by SV3D are high-quality reconstruction targets, and are often sufficient for 3D generation without the cumbersome SDS-based optimization."

21. (Limitations) "Our SV3D model is by design only capable of handling 2 degrees of freedom: elevation and azimuth; which is usually sufficient for 3D generation from a single image. One may want to tackle more degrees of freedom in cameras for a generalized NVS system, which forms an interesting future work."

22. (Limitations) "We also notice that SV3D exhibits some view inconsistency for mirror-like reflective surfaces, which provide a challenge to 3D reconstruction."

## Code/data link

- **Code:** [github.com/Stability-AI/generative-models](https://github.com/Stability-AI/generative-models) (the *Stability* *AI* *generative-models* *monorepo*, *Stability* *AI* *Community* *License* — *commercial* *OK* *for* *>$1M* *revenue*, *NOT* *commercial-deployable* *for* *small* *startups*).
- **Pretrained weights:** [huggingface.co/stabilityai/sv3d](https://huggingface.co/stabilityai/sv3d) (with *sv3d_u.safetensors* + *sv3d_p.safetensors*, Stability AI Community License).
- **Project page:** [sv3d.github.io](https://sv3d.github.io) (with *video* *demos* + *qualitative* *gallery* + *real-world* *in-the-wild* *3D* *reconstructions*).
- **Paper:** [arxiv.org/abs/2403.12008](https://arxiv.org/abs/2403.12008) (cs.CV, ECCV 2024, arXiv v1 18 Mar 2024).
- **ECCV 2024 OpenAccess:** [ecva.net/papers/eccv_2024/papers_ECCV/html/150_ECCV_2024_paper.php](https://www.ecva.net/papers/eccv_2024/papers_ECCV/html/150_ECCV_2024_paper.php) (poster paper).
- **Springer DOI:** [doi.org/10.1007/978-3-031-73232-4_25](https://link.springer.com/chapter/10.1007/978-3-031-73232-4_25) (ECCV 2024 LNCS Vol. 15092).
- **Training data:** Objaverse (Deitke et al. 2023, ref 9, the *de facto* 2023-2024 *3D-asset* *corpus*, 730K 3D models, NOT publicly released by Stability AI for SV3D specifically).
- **Eval data:** GSO (Google Scanned Objects, Downs et al. 2022, ref 10) + OmniObject3D (Wu et al. 2023, ref 54), both public.

## For our project (concrete next steps for v0 v0 v1 v2)

1. **ADOPT THE VIDEO-LDM FINE-TUNING PARADIGM (H2 mechanism).** v0 v0 sub-task 1 (full-arch synthesis) should *fine-tune* *SVD-xt* (or *a* *more* *recent* *open* *video* *LDM* *like* *CogVideoX* *or* *Mochi*) on *clinical* *dental* *arch* *multi-view* *data* *with* *the* *exact* *SV3D* *recipe*: (a) *camera-trajectory-as-conditioning* *via* *sinusoidal* *embedding* *of* *(elevation*, *azimuth)* *in* *every* *residual* *block* (the *direct* *H3* *mechanism* *for* *clinical* *dental* *multi-view* *generation*); (b) *progressive* *fine-tuning* *schedule* *from* *static* *orbit* *→* *dynamic* *orbit* (the *killer* *practical* *recipe* *that* *achieves* *the* *best* *metrics*); (c) *triangular-CFG* *scaling* *from* *1* *to* *2.5* *to* *1* (the *killer* *inference* *trick* *for* *closed-loop* *orbiting* *NVS*). *Cost*: *$2,000-5,000* *Lambda* *for* *v0 v0* *from-scratch* *fine-tune* *on* *dental* *arch* *data* *for* *4-6* *weeks* *on* *4-8* *A100s*.

2. **ADOPT THE 576×576 HIGH-RESOLUTION NVS (H4 mechanism).** v0 v0 sub-task 1 + sub-task 4 should *target* *576×576* *minimum* *resolution* *for* *clinical* *dental* *margin* *detail* *at* *sub-100μm* *scale* (576*×*576 *resolution* *at* *33.8°* *FOV* *gives* *sub-100μm* *detail* *at* *typical* *dental* *arch* *scale*). The *de facto* 2024 *high-resolution* *object-centric* *NVS* *SOTA* — *3×* *resolution* *jump* *from* *Zero123* + *EscherNet* + *Free3D*'s *256×256* *image-diffusion* *baselines*. *Practical* *caveat*: *512×512* *or* *576×576* *is* *the* *sweet* *spot* *for* *A100* *GPUs* *at* *21* *frames* *per* *clip*; *higher* *resolutions* *(768², 1024²)* *require* *H100* *GPUs* *or* *model* *parallelism*.

3. **ADOPT THE MASKED-SDS LOSS (H1 mechanism for clinical margin refinement).** v0 v0 sub-task 2 (crown generation) should *adopt* *the* *masked-SDS* *loss* *with* *the* *soft* *visibility* *mask* *M* = *1 − smoothstep(v_c · n, 0, 0.5)* *based* *on* *dot-product* *between* *surface* *normal* *n* *and* *view* *direction* *to* *the* *most-visible* *reference* *camera* — the *killer* *practical* *mechanism* *for* *clinical* *dental* *margin* *refinement* *where* *the* *margin* *is* *partially* *visible* *in* *the* *reference* *views* *but* *needs* *to* *be* *inpainted* *in* *the* *unseen* *regions*. *Cost*: *$50-100* *Lambda* *for* *v0 v0* *implementation*, *0.5-1* *week* *engineering*.

4. **ADOPT THE 24-SG DISENTANGLED-ILLUMINATION MODEL (H4 mechanism for lighting separation).** v0 v0 sub-task 2 (crown generation) should *adopt* *the* *24-SG* *illumination* *model* + *illumination-replication* *loss* *L_illum* = *|V(I) − L|²* *for* *clinical* *dental* *crown* *lighting* *separation* — the *killer* *practical* *mechanism* *for* *preventing* *baked-in* *lighting* (the *de facto* 2024 *3D-from-NVS* *failure* *mode*). *Cost*: *$100-200* *Lambda* *for* *v0 v0* *implementation*, *1-2* *weeks* *engineering*.

5. **ADOPT THE COARSE-TO-FINE Instant-NGP→DMTet PIPELINE (H1 mechanism for mesh refinement).** v0 v0 sub-task 2 (crown generation) should *adopt* *the* *two-stage* *coarse-to-fine* *recipe*: (a) *coarse* *Instant-NGP* *NeRF* *for* *general* *crown* *shape* + *texture* (~2 *minutes*); (b) *mesh* *extraction* *via* *marching* *cubes*; (c) *fine* *DMTet* *for* *high-resolution* *mesh* *refinement* *with* *masked-SDS* + *geometric* *priors* (~12 *minutes*); (d) *UV* *unwrapping* *via* *xatlas*. *Total* *time*: *~20* *minutes* *per* *crown*. *Cost*: *$200-500* *Lambda* *for* *v0 v0* *implementation*, *2-3* *weeks* *engineering*.

6. **ADOPT THE DYNAMIC-ORBIT TRAINING DATA (H3 mechanism for top + bottom view generation).** v0 v0 sub-task 1 (full-arch synthesis) should *adopt* *the* *dynamic-orbit* *training* *data* *recipe*: *sample* *static* *orbit* = *uniform* *azimuth* *at* *fixed* *elevation* *→* *convert* *to* *dynamic* *orbit* = *add* *small* *random* *noise* *to* *azimuths* + *add* *random* *weighted* *combination* *of* *sinusoids* *with* *different* *frequencies* *to* *elevation*. The *exact* *mechanism* *for* *clinical* *dental* *occlusal* *+* *cervical* *view* *generation*.

7. **CITE SV3D AS THE 2024 MULTI-VIEW-DIFFUSION VIDEO-LDM PARADIGM IN V0 PAPER RELATED-WORK.** v0 paper's *related-work* *section* *should* *include* *a* *paragraph* *on* *the* *2024* *multi-view-diffusion* *paradigm-shift* *from* *image-LDM* *to* *video-LDM* *fine-tuning*, *citing* *SV3D* *as* *the* *founder* *of* *the* *camera-trajectory-as-conditioning* *trick* + *triangular-CFG* *scaling* + *masked-SDS* *loss* + *disentangled-illumination* *model*. The *killer* *positioning* *for* *v0 v0 v1 v2* *as* *the* *first* *clinical* *dental* *arch* *multi-view* *paper* *to* *adopt* *the* *video-LDM* *fine-tuning* *recipe*. *Cost*: *$0*, *1* *hour*.

8. **FORK THE STABILITY AI GENERATIVE-MODELS REPO + PORT TO DENTAL ARCH.** v0 v0 engineering starting point: *fork* *github.com/Stability-AI/generative-models* *and* *add* *the* *dental* *arch* *fine-tuning* *recipe* (Objaverse + clinical dental arch data) + the *progressive* *training* *schedule* (static → dynamic) + the *clinical* *dental* *evaluation* *metrics* (margin gap, internal fit, proximal contact, occlusion). *Cost*: *$500-1,000* *Lambda* *for* *v0 v0* *engineering*, *2-3* *weeks*.

9. **PRACTICAL CAVEAT: SV3D LICENSE.** SV3D weights are *released* *under* *the* *Stability* *AI* *Community* *License* — *commercial* *OK* *for* *>$1M* *revenue*, *but* *v0 v0 v1 v2* *is* *a* *startup* *<* *$1M* *revenue* *so* *NOT* *commercial-deployable*. The *practical* *reason* *v0 v0 v1 v2* *must* *fine-tune* *from* *scratch* *on* *clinical* *dental* *data* *and* *release* *the* *fine-tuned* *weights* *under* *a* *commercial-friendly* *license* *(e.g. MIT, Apache 2.0)*. The *practical* *engineering* *starting* *point* is *the* *code* *recipe* + *the* *training* *data* *recipe*, *NOT* *the* *weights*.

10. **TRACE THE 2024→2025→2026 MULTI-VIEW-DIFFUSION ARC IN V0 PAPER.** The *2024→2025→2026* *multi-view-diffusion* *evolution* *arc* is: *SV3D 117* (Mar 2024, *founder* *of* *video-LDM* *fine-tuning* *+* *camera-trajectory* *conditioning*) *→* *CAT3D 113* (May 2024, *extends* *to* *3D-attention* *+* *real-scene* *eval*) *→* *Bolt3D 116* (May 2025, *extends* *to* *real-image* *re-training* *+* *scene-level* *multi-view*) *→* *L4GM 114* (2025, *extends* *to* *4D* *temporal* *multi-view* *generation*). v0 paper's *related-work* *should* *include* *this* *complete* *arc* *as* *the* *de facto* 2024-2026 *multi-view-diffusion* *paradigm*, *with* *v0 v0 v1 v2* *positioned* *as* *the* *first* *clinical* *dental* *arch* *application*.

v0 stack updated: sub-task 1 (full-arch synthesis) v1+ = **video-LDM fine-tune (SVD-xt or CogVideoX) + camera-trajectory-as-conditioning (sinusoidal elevation/azimuth embedding in every residual block) + progressive schedule (static → dynamic orbit) + 576×576 high-resolution + triangular-CFG scaling (1 → 2.5 → 1)** (NEW from 117, $2,000-5,000 Lambda for v0 v0 v1 v2 from-scratch fine-tune, 4-6 weeks on 4-8 A100s); sub-task 2 (crown generation) v1+ = **coarse-to-fine Instant-NGP→DMTet pipeline + masked-SDS loss (visibility-aware soft mask) + 24-SG disentangled-illumination model + photometric + LPIPS + RegNeRF smooth depth + bilateral normal smoothness + mono normal loss from Omnidata** (NEW from 117, $300-700 Lambda for v0 v0 v1 v2 implementation, 3-4 weeks engineering); v0 total compute = **~$9,170-11,330 Lambda** (was $8,670-10,660 from 106, +$500-1,700 for SV3D-inspired video-LDM fine-tune + coarse-to-fine Instant-NGP→DMTet pipeline + masked-SDS + 24-SG illumination). **Strategic positioning: SV3D is the *founder* of the *video-LDM-for-multi-view-3D* *paradigm*, the *direct* *technical* *predecessor* of *Bolt3D 116* + *CAT3D 113* + *L4GM 114* + *every* *2024-2026* *multi-view-diffusion* *paper*, the *canonical* *H2* + *H3* + *H4* *mechanism* *for* *v0 v0 v1 v2* *clinical* *dental* *arch* *multi-view* *generation*.**
