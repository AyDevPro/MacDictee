// macdictee-audio : capture le son du Mac (visios, vidéos… tout sauf MacDictée) et l'écrit sur stdout
// en PCM float32 mono 16 kHz. MacDictée le lance pendant une réunion et le lit en continu ; fermer stdin
// (ou quitter MacDictée) arrête la capture. Sur stderr : « ready », « mode … », « error: … ».
//
// macOS 14.2+ : Core Audio process tap, autorisation « Enregistrement des sons du système uniquement »
//   (comme Wispr Flow ; Info.plist de MacDictée : NSAudioCaptureUsageDescription).
// Avant : ScreenCaptureKit, autorisation « Enregistrement de l'écran et des sons du système ».
// Compilation : swiftc -O -target arm64-apple-macos13.0 -o macdictee-audio audio_capture.swift
import AVFoundation
import CoreAudio
import CoreMedia
import Foundation
import ScreenCaptureKit

let stdout = FileHandle.standardOutput
func log(_ message: String) { FileHandle.standardError.write((message + "\n").data(using: .utf8)!) }
let outFormat = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: 16000, channels: 1, interleaved: false)!

// ---------------------------------------------------------------- Core Audio process tap (macOS 14.2+)
func property<T>(_ object: AudioObjectID, _ selector: AudioObjectPropertySelector, _ initial: T,
                 qualifier: UInt32? = nil) -> T? {
    var address = AudioObjectPropertyAddress(mSelector: selector, mScope: kAudioObjectPropertyScopeGlobal,
                                             mElement: kAudioObjectPropertyElementMain)
    var value = initial
    var size = UInt32(MemoryLayout<T>.size)
    var status: OSStatus
    if var q = qualifier {
        status = AudioObjectGetPropertyData(object, &address, UInt32(MemoryLayout<UInt32>.size), &q, &size, &value)
    } else {
        status = AudioObjectGetPropertyData(object, &address, 0, nil, &size, &value)
    }
    return status == noErr ? value : nil
}

@available(macOS 14.2, *)
final class TapCapture {
    private var tapID = AudioObjectID(kAudioObjectUnknown)
    private var aggregateID = AudioObjectID(kAudioObjectUnknown)
    private var procID: AudioDeviceIOProcID?
    private var converter: AVAudioConverter?

    func start() throws {
        // on n'enregistre ni MacDictée ni ce module (sons « Tink » / « Pop » de la dictée)
        let excluded = [getpid(), getppid()].compactMap {
            property(AudioObjectID(kAudioObjectSystemObject), kAudioHardwarePropertyTranslatePIDToProcessObject,
                     AudioObjectID(kAudioObjectUnknown), qualifier: UInt32(bitPattern: $0))
        }.filter { $0 != kAudioObjectUnknown }
        let description = CATapDescription(stereoGlobalTapButExcludeProcesses: excluded)
        description.uuid = UUID()
        description.isPrivate = true
        description.muteBehavior = .unmuted
        try check(AudioHardwareCreateProcessTap(description, &tapID), "création du tap")

        guard var format = property(tapID, kAudioTapPropertyFormat, AudioStreamBasicDescription()),
              let inFormat = AVAudioFormat(streamDescription: &format) else { throw failure("format du tap") }
        converter = AVAudioConverter(from: inFormat, to: outFormat)
        converter?.downmix = true

        guard let output = property(AudioObjectID(kAudioObjectSystemObject), kAudioHardwarePropertyDefaultOutputDevice,
                                    AudioObjectID(kAudioObjectUnknown)),
              let outputUID = property(output, kAudioDevicePropertyDeviceUID, "" as CFString) else {
            throw failure("sortie audio par défaut")
        }
        let aggregate: [String: Any] = [
            kAudioAggregateDeviceNameKey: "MacDictée (son du Mac)",
            kAudioAggregateDeviceUIDKey: UUID().uuidString,
            kAudioAggregateDeviceMainSubDeviceKey: outputUID as String,
            kAudioAggregateDeviceIsPrivateKey: true,
            kAudioAggregateDeviceIsStackedKey: false,
            kAudioAggregateDeviceTapAutoStartKey: true,
            kAudioAggregateDeviceSubDeviceListKey: [[kAudioSubDeviceUIDKey: outputUID as String]],
            kAudioAggregateDeviceTapListKey: [[kAudioSubTapDriftCompensationKey: true,
                                               kAudioSubTapUIDKey: description.uuid.uuidString]],
        ]
        try check(AudioHardwareCreateAggregateDevice(aggregate as CFDictionary, &aggregateID), "périphérique agrégé")
        try check(AudioDeviceCreateIOProcIDWithBlock(&procID, aggregateID, nil) { [weak self] _, input, _, _, _ in
            self?.forward(input, inFormat)
        }, "lecture audio")
        try check(AudioDeviceStart(aggregateID, procID), "démarrage")
        log("mode tap \(Int(inFormat.sampleRate)) Hz, \(inFormat.channelCount) canal(aux)")
    }

    private func forward(_ input: UnsafePointer<AudioBufferList>, _ inFormat: AVAudioFormat) {
        guard let converter, let inBuffer = AVAudioPCMBuffer(pcmFormat: inFormat, bufferListNoCopy: input, deallocator: nil),
              inBuffer.frameLength > 0 else { return }
        let capacity = AVAudioFrameCount(Double(inBuffer.frameLength) * 16000 / inFormat.sampleRate) + 32
        guard let outBuffer = AVAudioPCMBuffer(pcmFormat: outFormat, frameCapacity: capacity) else { return }
        var consumed = false
        converter.convert(to: outBuffer, error: nil) { _, status in
            if consumed { status.pointee = .noDataNow; return nil }
            consumed = true
            status.pointee = .haveData
            return inBuffer
        }
        if let channel = outBuffer.floatChannelData, outBuffer.frameLength > 0 {
            stdout.write(Data(bytes: channel[0], count: Int(outBuffer.frameLength) * 4))
        }
    }

    func stop() {
        if aggregateID != kAudioObjectUnknown {
            AudioDeviceStop(aggregateID, procID)
            if let procID { AudioDeviceDestroyIOProcID(aggregateID, procID) }
            AudioHardwareDestroyAggregateDevice(aggregateID)
        }
        if tapID != kAudioObjectUnknown { AudioHardwareDestroyProcessTap(tapID) }
    }

    private func check(_ status: OSStatus, _ step: String) throws {
        if status != noErr { throw failure("\(step) (code \(status))") }
    }
}

func failure(_ message: String) -> NSError {
    NSError(domain: "macdictee", code: 1, userInfo: [NSLocalizedDescriptionKey: message])
}

// ---------------------------------------------------------------- ScreenCaptureKit (macOS 13 à 14.1)
final class ScreenCapture: NSObject, SCStreamOutput, SCStreamDelegate {
    private var stream: SCStream?

    func start() async throws {
        let content = try await SCShareableContent.excludingDesktopWindows(false, onScreenWindowsOnly: false)
        guard let display = content.displays.first else { throw failure("aucun écran") }
        let config = SCStreamConfiguration()
        config.capturesAudio = true
        config.sampleRate = 16000
        config.channelCount = 1
        config.excludesCurrentProcessAudio = true
        // ScreenCaptureKit capture toujours un flux vidéo : on le réduit au strict minimum
        config.width = 2
        config.height = 2
        config.minimumFrameInterval = CMTime(value: 1, timescale: 1)
        let filter = SCContentFilter(display: display, excludingApplications: [], exceptingWindows: [])
        let stream = SCStream(filter: filter, configuration: config, delegate: self)
        try stream.addStreamOutput(self, type: .audio, sampleHandlerQueue: DispatchQueue(label: "audio"))
        try stream.addStreamOutput(self, type: .screen, sampleHandlerQueue: DispatchQueue(label: "screen"))
        try await stream.startCapture()
        self.stream = stream
        log("mode écran")
    }

    func stream(_ stream: SCStream, didOutputSampleBuffer sampleBuffer: CMSampleBuffer, of type: SCStreamOutputType) {
        guard type == .audio, sampleBuffer.isValid else { return }
        // float32 non entrelacé : un tampon par canal ; on garde le premier (mono demandé)
        try? sampleBuffer.withAudioBufferList { buffers, _ in
            guard let first = buffers.first, let data = first.mData else { return }
            stdout.write(Data(bytes: data, count: Int(first.mDataByteSize)))
        }
    }

    func stream(_ stream: SCStream, didStopWithError error: Error) {
        log("error: \(error.localizedDescription)")
        exit(2)
    }
}

// ---------------------------------------------------------------- lancement
var stopCapture: () -> Void = {}
Task {
    do {
        if #available(macOS 14.2, *), !CommandLine.arguments.contains("--screen") {
            let tap = TapCapture()
            stopCapture = { tap.stop() }
            try tap.start()
        } else {
            try await ScreenCapture().start()
        }
        log("ready")
    } catch {
        log("error: \(error.localizedDescription)")  // le plus souvent : autorisation refusée
        exit(1)
    }
}
// MacDictée ferme notre stdin pour arrêter la capture
DispatchQueue.global().async {
    _ = FileHandle.standardInput.readDataToEndOfFile()
    stopCapture()
    exit(0)
}
dispatchMain()
