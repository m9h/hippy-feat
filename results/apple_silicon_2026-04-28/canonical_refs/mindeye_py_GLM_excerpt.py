            -ref {fmriprep_boldref} \
            -out {final_vol} \
            -init {current_tr_to_orig_ses} \
            -applyxfm")  # apply combined transformation matrix to the current TR

        os.system(f"rm -r {mc}.mat")
        imgs.append(get_data(final_vol))

        if current_label not in ('blank', 'blank.jpg'):
            events_df = events_df.copy()
            events_df['onset'] = events_df['onset'].astype(float)

            run_start_time = events_df['onset'].iloc[0]
            events_df = events_df.copy()
            events_df['onset'] -= run_start_time

            cropped_events = events_df[events_df.onset <= TR*tr_length]
            cropped_events = cropped_events.copy()
            cropped_events.loc[:, 'trial_type'] = np.where(cropped_events['trial_number'] == stimulus_trial_counter, "probe", "reference")
            cropped_events = cropped_events.drop(columns=['is_correct', 'image_name', 'response_time', 'trial_number'])

            # collect all of the images at each TR into a 4D time series
            img = np.rollaxis(np.array(imgs),0,4)
            img = new_img_like(fmriprep_boldref_nib,img,copy_header=True)
            # run the model with mc_params confounds to motion correct
            lss_glm = FirstLevelModel(t_r=tr_length,slice_time_ref=0,hrf_model='glover',
                        drift_model='cosine', drift_order=1,high_pass=0.01,mask_img=union_mask_img,
                        signal_scaling=False,smoothing_fwhm=None,noise_model='ar1',
                        n_jobs=-1,verbose=-1,memory_level=1,minimize_memory=True)
            
            lss_glm.fit(run_imgs=img, events=cropped_events, confounds = pd.DataFrame(np.array(mc_params)))
            dm = lss_glm.design_matrices_[0]
            # get the beta map and mask it
            beta_map = lss_glm.compute_contrast("probe", output_type="effect_size")
            beta_map_np = beta_map.get_fdata()
            beta_map_np = fast_apply_mask(target=beta_map_np,mask=union_mask_img.get_fdata())
            all_betas.append(beta_map_np)
            print('all_betas shape:', np.array(all_betas).shape)
            
            if current_label not in shown_filenames.keys():
                shown_filenames[current_label] = [len(all_betas)]
                is_repeat = False
            else:
                shown_filenames[current_label].append(len(all_betas))
                is_repeat = True
                print(f"The following image is a repeat!\n{shown_filenames[current_label]}")

            if "MST_pairs" in current_label and run_num >= 2:
                mst_trial_counter += 1
                if mst_trial_counter in mst_recon_points:
                    correct_image_index = np.where(current_label == vox_image_names)[0][0]  # using the first occurrence based on image name, assumes that repeated images are identical (which they should be)
                    z_mean = np.mean(np.array(all_betas), axis=0)
                    z_std = np.std(np.array(all_betas), axis=0)
                    if is_repeat:
                        beta_repeat_idxs = shown_filenames[current_label]
                        assert len(beta_repeat_idxs) > 1  # this image has been shown more than once
                        betas_repeats = []
                        for b in beta_repeat_idxs:
                            print(f"Averaging over {len(beta_repeat_idxs)} repeats")
                            # re-z-score the older betas in addition to the newest beta since we have more data to z-score with
                            tmp = ((np.array(all_betas) - z_mean) / (z_std + 1e-6))[b-1]
                            betas_repeats.append(tmp)
                        betas = np.mean(np.array(betas_repeats), axis=0)  # average beta patterns over all available repeats
                    else:
                        betas = ((np.array(all_betas) - z_mean) / (z_std + 1e-6))[-1]  # use only the beta pattern from the most recent image
                    betas = betas[np.newaxis, np.newaxis, :]
                    betas_tt = torch.Tensor(betas).to("cpu")
                    reconsTR, clipvoxelsTR = do_reconstructions(betas_tt)
                    if clipvoxelsTR is None:
                        with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.float16):
                            voxel = betas_tt
                            voxel = voxel.to(device)
                            assert voxel.shape[1] == 1
                            voxel_ridge = model.ridge(voxel[:,[-1]],0) # 0th index of subj_list
                            backbone0, clip_voxels0, blurry_image_enc0 = model.backbone(voxel_ridge)
                            clip_voxels = clip_voxels0
                            backbone = backbone0
                            blurry_image_enc = blurry_image_enc0[0]
                            clipvoxelsTR = clip_voxels.cpu()

                    values_dict = get_top_retrievals(clipvoxelsTR, all_images=images[MST_idx], total_retrievals=5)
