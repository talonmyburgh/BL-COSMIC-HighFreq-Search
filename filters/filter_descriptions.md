Filters run on the hits from Coherent beams on COSMIC targest (30M closest stars in GAIA catalogue):
1. Grouped by source. Grouped by frequency. Rejects groups where all hits have 0 drift rate
2. Grouped by source. Grouped by frequency. Rejects groups where all hits have drift rate < 0.25Hz/s
3. Grouped by source. Grouped by frequency. Rejects groups where all hits have drift rate < 2Hz/s
4. Grouped by source. Grouped by frequency. Rejects groups with more than one frequency
5. Grouped by freuqency. Rejects groups with more than one frequency.
6. Rejects all hits with zero drift rate.
7. Rejects all hits with SNR < 10
8. Rejects all hits with SNR > 100
9. Rejects hits within 2hz of another hit
10. Rejects hits within 10hz of another hit
11. Flag hits which have another hit where it looks like they're drifting to
12. For timestamps which have a num_timesteps < 16, reject hits with SNR < 15

Motivation:
1. Improve sensitivity to rejecting signals which have consistently zero drift rate (so are not astrophysical but instead originate on the Earth itself which is why they have 0 relative acceleration). COSMIC's sensitivity is to signals drifting 0.25Hz/s or more. 
2. Also reject signals which might, because of the FFT, get moved one frequency bin or another to the side but later get detected at the same frequency (didn't actually drift).
3. Also reject signals which might modulate frequency slightly within a band and thus get detected with a slight drift rate.
4. The odds of two hits being seen at exactly the same frequency are pretty low in this dataset (still needs further analysis but each hit has ~100 bins to itself at worst) so this hopefully gets rid of persistent RFI on the sky
5. Previous filters only filtered out persistent RFI which was coming from the same source (direction on the sky) but this will also filter out RFI which is seen across multiple places on the sky
6. We expect astrophysical signals to drift in the frequency domain due to a nonzero relative acceleration with the telescope
7. Hits with SNRs this low are typically spurious signals coming from noise. Addiional work is needed to improve the detection pipeline if we want to be sensitive to signals this weak.
8. We expect technosignatures to be fairly weak if they're coming from space and losing power to the inverse square law over huge distances. If a signal has a really high power, it's likely from a source of RFI close by
9. The FFT can shift signals a bin in the frequency domain pretty easily. If a signal is persistent without drifting, it's likely RFI
10. Same as above but also sensitive to signals which modulate frequency slightly within the same band.
11. If signals really are drifting, then we should see them having drifted in follow-up observations in the direction we originally detected them. Highlight and pass on these hits. Reject those with no connection. We can do this kind of search because sources are observed more than once. It's not a perfect filter and could account for base cases like these better. Should be updated before this method is used in the future.
12. The fewer timesteps observed, the more likely seticore is to detect a spurious signal as a hit even if there's nothing there. Impose a more stringent SNR requirement on observations with fewer timesteps. Exact values require further statistical analysis to better understand what they should be. These were naive guesses.

Additional information:
Many hits are coming from signals which only come from one antenna at a time and have very low drift rate (below 2Hz/s) but are usually pretty strong. We think these signals are coming from either issues with how the data is processed or RFI from the electronics of the antennas. We don't think it's astrophysical because not all of the antennas see it (imagine pointing 27 cameras at the same object and only one of them sees something out of the ordinary). Filters 1-6 and 9-10 are designed to get rid of these signals.

Many hits are also coming from noise the seticore detection pipeline picks up as a technosignature. These typically have low power and signal to noise ratio, although they can have arbitrary drift rates. The fewer data points (in the time domain) or num_timesteps collected and searched for technosignatures, the more likely these spurious hits are to occur. Filters 7 and 12 are sensitive to these spurious signals.


